import os
from dataclasses import dataclass

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit

from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
from TTS.tts.layers.xtts.stream_generator import init_stream_support
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager, LanguageManager
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.io import load_fsspec

init_stream_support()


def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    """
    Load and preprocess audio for inference.
    """
    audio, lsr = torchaudio.load(audiopath)

    # stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    audio.clip_(-1, 1)
    return audio


def pad_or_truncate(t, length):
    """
    Ensure a given tensor t has a specified sequence length by either padding it with zeros or clipping it.

    Args:
        t (torch.Tensor): The input tensor to be padded or truncated.
        length (int): The desired length of the tensor.

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp


@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000


@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_code_stride_len: int = 1024

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True


class Xtts(BaseTTS):
    """ⓍTTS model implementation."""

    def __init__(self, config: Coqpit):
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        self.gpt_batch_size = self.args.gpt_batch_size
        self.tokenizer = VoiceBpeTokenizer()
        self.init_models()

    def init_models(self):
        """Initialize the models."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                code_stride_len=self.args.gpt_code_stride_len,
            )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            decoder_input_dim=self.args.decoder_input_dim,
            d_vector_dim=self.args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio."""
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]

        mel = wav_to_mel_cloning(
            audio,
            mel_norms=self.mel_stats.cpu(),
            n_fft=4096,
            hop_length=1024,
            win_length=4096,
            power=2,
            normalized=False,
            sample_rate=22050,
            f_min=0,
            f_max=8000,
            n_mels=80,
        )
        cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        """Generate the speaker embedding."""
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @torch.inference_mode()
    def get_conditioning_latents(self, audio_path, max_ref_length=30, gpt_cond_len=6, gpt_cond_chunk_len=6):
        """Get the conditioning latents for the GPT model from the given audio."""
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        audios = []
        for file_path in audio_paths:
            audio = load_audio(file_path, self.config.sample_rate)
            audio = audio[:, : self.config.sample_rate * max_ref_length].to(self.device)

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, self.config.sample_rate)
            audios.append(audio)

        # merge all the audios and compute the latents for the GPT
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, self.config.sample_rate, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )

        return gpt_cond_latents, speaker_embedding

    @torch.inference_mode()
    def batch_inference(
        self,
        text_batch,  # Expecting a list of sentences (batch)
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        """Batch inference for multiple texts."""
        language = language.split("-")[0]  # Remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)

        # Tokenizing and padding the text in batch mode
        text_tokens = []
        lens = []
        for text in text_batch:
            text = text.strip().lower()
            text_token = torch.IntTensor(self.tokenizer.encode(text, lang=language)).unsqueeze(0)
            lens.append(text_token.shape[1])
            text_tokens.append(text_token)

        max_text_len = max(lens)
        text_padded = torch.zeros(len(text_batch), max_text_len, dtype=torch.int64).to(self.device)
        for i, token in enumerate(text_tokens):
            text_padded[i, :lens[i]] = token

        # Repeat latents for batch size
        xg = gpt_cond_latent.repeat(len(text_batch), 1, 1)
        xse = speaker_embedding.repeat(len(text_batch), 1, 1)

        with torch.no_grad():
            # Generate GPT codes in batch
            gpt_codes = self.gpt.generate(
                cond_latents=xg,
                text_inputs=text_padded,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=1,  # Adjust to support multiple outputs per input if needed
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                **hf_generate_kwargs,
            )

            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=self.device
            )

            text_len = torch.tensor(lens, device=self.device)
            gpt_latents = self.gpt(
                text_padded,
                text_len,
                gpt_codes,
                expected_output_len,
                cond_latents=xg,
                return_latent=True,
            )

            if length_scale != 1.0:
                gpt_latents = F.interpolate(
                    gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                ).transpose(1, 2)

            # Decode generated latents into audio
            wavs = self.hifigan_decoder(gpt_latents, g=xse).cpu().squeeze()

        return {
            "wav": wavs.cpu().unsqueeze(1),  # Return the batch of generated audio
            "gpt_latents": gpt_latents.cpu().numpy(),
            "speaker_embedding": speaker_embedding,
        }

    @torch.inference_mode()
    def full_batch_inference(
        self,
        text_batch,  # List of sentences (batch)
        ref_audio_path,  # Path to reference audio
        language,
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        **hf_generate_kwargs,
    ):
        """Batch inference for multiple sentences with reference audio."""
        gpt_cond_latent, speaker_embedding = self.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs,
        )

        return self.batch_inference(
            text_batch,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **hf_generate_kwargs,
        )
