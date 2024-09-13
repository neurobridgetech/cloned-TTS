import tempfile
import warnings
from pathlib import Path
from typing import Union, List

import numpy as np
from torch import nn

from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.config import load_config


class TTS(nn.Module):
    """TODO: Add voice conversion and Capacitron support."""

    def __init__(
        self,
        model_name: str = "",
        model_path: str = None,
        config_path: str = None,
        vocoder_path: str = None,
        vocoder_config_path: str = None,
        progress_bar: bool = True,
        gpu=False,
    ):
        """🐸TTS python interface that allows to load and use the released models."""

        super().__init__()
        self.manager = ModelManager(models_file=self.get_models_file_path(), progress_bar=progress_bar, verbose=False)
        self.config = load_config(config_path) if config_path else None
        self.synthesizer = None
        self.voice_converter = None
        self.model_name = ""
        if gpu:
            warnings.warn("`gpu` will be deprecated. Please use `tts.to(device)` instead.")

        if model_name is not None and len(model_name) > 0:
            if "tts_models" in model_name:
                self.load_tts_model_by_name(model_name, gpu)
            elif "voice_conversion_models" in model_name:
                self.load_vc_model_by_name(model_name, gpu)
            else:
                self.load_model_by_name(model_name, gpu)

        if model_path:
            self.load_tts_model_by_path(
                model_path, config_path, vocoder_path=vocoder_path, vocoder_config=vocoder_config_path, gpu=gpu
            )

    @property
    def models(self):
        return self.manager.list_tts_models()

    @property
    def is_multi_speaker(self):
        if hasattr(self.synthesizer.tts_model, "speaker_manager") and self.synthesizer.tts_model.speaker_manager:
            return self.synthesizer.tts_model.speaker_manager.num_speakers > 1
        return False

    @property
    def is_multi_lingual(self):
        if (
            isinstance(self.model_name, str)
            and "xtts" in self.model_name
            or self.config
            and ("xtts" in self.config.model or len(self.config.languages) > 1)
        ):
            return True
        if hasattr(self.synthesizer.tts_model, "language_manager") and self.synthesizer.tts_model.language_manager:
            return self.synthesizer.tts_model.language_manager.num_languages > 1
        return False

    @property
    def speakers(self):
        if not self.is_multi_speaker:
            return None
        return self.synthesizer.tts_model.speaker_manager.speaker_names

    @property
    def languages(self):
        if not self.is_multi_lingual:
            return None
        return self.synthesizer.tts_model.language_manager.language_names

    @staticmethod
    def get_models_file_path():
        return Path(__file__).parent / ".models.json"

    def list_models(self):
        return ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False)

    def download_model_by_name(self, model_name: str):
        model_path, config_path, model_item = self.manager.download_model(model_name)
        if "fairseq" in model_name or (model_item is not None and isinstance(model_item["model_url"], list)):
            # return model directory if there are multiple files
            # we assume that the model knows how to load itself
            return None, None, None, None, model_path
        if model_item.get("default_vocoder") is None:
            return model_path, config_path, None, None, None
        vocoder_path, vocoder_config_path, _ = self.manager.download_model(model_item["default_vocoder"])
        return model_path, config_path, vocoder_path, vocoder_config_path, None

    def load_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the 🐸TTS models by name."""
        self.load_tts_model_by_name(model_name, gpu)

    def load_vc_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the voice conversion models by name."""
        self.model_name = model_name
        model_path, config_path, _, _, _ = self.download_model_by_name(model_name)
        self.voice_converter = Synthesizer(vc_checkpoint=model_path, vc_config=config_path, use_cuda=gpu)

    def load_tts_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of 🐸TTS models by name."""
        self.synthesizer = None
        self.model_name = model_name

        model_path, config_path, vocoder_path, vocoder_config_path, model_dir = self.download_model_by_name(model_name)

        # init synthesizer
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config_path,
            encoder_checkpoint=None,
            encoder_config=None,
            model_dir=model_dir,
            use_cuda=gpu,
        )

    def load_tts_model_by_path(
        self, model_path: str, config_path: str, vocoder_path: str = None, vocoder_config: str = None, gpu: bool = False
    ):
        """Load a model from a path."""
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config,
            encoder_checkpoint=None,
            encoder_config=None,
            use_cuda=gpu,
        )

    def _check_arguments(
        self,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
        **kwargs,
    ) -> None:
        """Check if the arguments are valid for the model."""
        if self.is_multi_speaker and (speaker is None and speaker_wav is None):
            raise ValueError("Model is multi-speaker but no `speaker` is provided.")
        if self.is_multi_lingual and language is None:
            raise ValueError("Model is multi-lingual but no `language` is provided.")

    def tts(
        self,
        text: Union[str, List[str]],
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text or batch of texts to speech."""
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, emotion=emotion, speed=speed, **kwargs)

        if isinstance(text, list):  # Batch processing
            wavs = self.synthesizer.tts_batch(
                text_batch=text,
                speaker_name=speaker,
                language_name=language,
                speaker_wav=speaker_wav,
                split_sentences=split_sentences,
                **kwargs,
            )
            return wavs

        # Single sentence processing
        wav = self.synthesizer.tts(
            text=text,
            speaker_name=speaker,
            language_name=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        return wav

    def tts_to_file(
        self,
        text: Union[str, List[str]],
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = 1.0,
        pipe_out=None,
        file_path: Union[str, List[str]] = "output.wav",
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text or batch of texts to speech and save to file."""
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, **kwargs)

        if isinstance(text, list):  # Batch processing
            wavs = self.tts(
                text=text,
                speaker=speaker,
                language=language,
                speaker_wav=speaker_wav,
                split_sentences=split_sentences,
                **kwargs,
            )

            if isinstance(file_path, list) and len(file_path) == len(wavs):
                for wav, path in zip(wavs, file_path):
                    self.synthesizer.save_wav(wav=wav, path=path, pipe_out=pipe_out)
            else:
                raise ValueError("The number of file paths should match the number of input texts in batch mode.")
            return file_path

        # Single sentence processing
        wav = self.tts(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        self.synthesizer.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)
        return file_path

    def voice_conversion(
        self,
        source_wav: str,
        target_wav: str,
    ):
        """Voice conversion with FreeVC. Convert source wav to target speaker."""
        wav = self.voice_converter.voice_conversion(source_wav=source_wav, target_wav=target_wav)
        return wav

    def voice_conversion_to_file(
        self,
        source_wav: str,
        target_wav: str,
        file_path: str = "output.wav",
    ):
        """Voice conversion with FreeVC. Convert source wav to target speaker."""
        wav = self.voice_conversion(source_wav=source_wav, target_wav=target_wav)
        save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)
        return file_path

    def tts_with_vc(
        self,
        text: str,
        language: str = None,
        speaker_wav: str = None,
        speaker: str = None,
        split_sentences: bool = True,
    ):
        """Convert text to speech with voice conversion."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            self.tts_to_file(
                text=text, speaker=speaker, language=language, file_path=fp.name, split_sentences=split_sentences
            )
        if self.voice_converter is None:
            self.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")
        wav = self.voice_converter.voice_conversion(source_wav=fp.name, target_wav=speaker_wav)
        return wav

    def tts_with_vc_to_file(
        self,
        text: str,
        language: str = None,
        speaker_wav: str = None,
        file_path: str = "output.wav",
        speaker: str = None,
        split_sentences: bool = True,
    ):
        """Convert text to speech with voice conversion and save to file."""
        wav = self.tts_with_vc(
            text=text, language=language, speaker_wav=speaker_wav, speaker=speaker, split_sentences=split_sentences
        )
        save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)
