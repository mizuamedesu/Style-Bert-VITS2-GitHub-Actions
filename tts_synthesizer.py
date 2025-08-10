import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import numpy as np
from numpy.typing import NDArray

import torch
from scipy.io import wavfile

from config import get_config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

try:
    from style_bert_vits2.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TTSSynthesizer:
    
    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        self.config = get_config()
        
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_dir is None:
            model_dir = self.config.assets_root
        self.model_dir = Path(model_dir)
        
        pyopenjtalk.initialize_worker()
        update_dict()
        self._preload_bert_models()
        
        self.model_holder = TTSModelHolder(self.model_dir, self.device)
        if len(self.model_holder.model_names) == 0:
            logger.error(f"Models not found in {self.model_dir}.")
            raise ValueError(f"No models found in {self.model_dir}")
            
        self.loaded_models = self._load_models()
        logger.info(f"Loaded {len(self.loaded_models)} models")
        
    def _preload_bert_models(self):
        logger.info("Preloading BERT models...")
        bert_models.load_model(Languages.JP)
        bert_models.load_tokenizer(Languages.JP)
        bert_models.load_model(Languages.EN)
        bert_models.load_tokenizer(Languages.EN)
        bert_models.load_model(Languages.ZH)
        bert_models.load_tokenizer(Languages.ZH)
        
    def _load_models(self) -> List[TTSModel]:
        loaded_models = []
        for model_name, model_paths in self.model_holder.model_files_dict.items():
            model = TTSModel(
                model_path=model_paths[0],
                config_path=self.model_holder.root_dir / model_name / "config.json",
                style_vec_path=self.model_holder.root_dir / model_name / "style_vectors.npy",
                device=self.device,
            )
            loaded_models.append(model)
        return loaded_models
    
    def synthesize(
        self,
        text: str,
        model_name: Optional[str] = None,
        model_id: int = 0,
        speaker_name: Optional[str] = None,
        speaker_id: int = 0,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noisew: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        language: Languages = Languages.JP,
        auto_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        reference_audio_path: Optional[str] = None,
    ) -> Tuple[int, NDArray[np.int16]]:
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        if model_name:
            model_ids = [i for i, x in enumerate(self.model_holder.models_info) 
                        if x.name == model_name]
            if not model_ids:
                raise ValueError(f"Model '{model_name}' not found")
            if len(model_ids) > 1:
                raise ValueError(f"Model name '{model_name}' is ambiguous")
            model_id = model_ids[0]
        elif model_id >= len(self.loaded_models):
            raise ValueError(f"Model ID {model_id} not found")
            
        model = self.loaded_models[model_id]
        
        if speaker_name is not None:
            if speaker_name not in model.spk2id.keys():
                raise ValueError(f"Speaker '{speaker_name}' not found")
            speaker_id = model.spk2id[speaker_name]
        elif speaker_id not in model.id2spk.keys():
            raise ValueError(f"Speaker ID {speaker_id} not found")
            
        if style and style not in model.style2id.keys():
            raise ValueError(f"Style '{style}' not found")
            
        sr, audio = model.infer(
            text=text,
            language=language,
            speaker_id=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )
        
        if hasattr(logger, 'success'):
            logger.success(f"Audio synthesized successfully: {len(audio)} samples at {sr}Hz")
        else:
            logger.info(f"Audio synthesized successfully: {len(audio)} samples at {sr}Hz")
        return sr, audio
    
    def save_audio(self, sr: int, audio: NDArray[np.int16], output_path: str):
        wavfile.write(output_path, sr, audio)
        logger.info(f"Audio saved to: {output_path}")
        
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for model_id, model in enumerate(self.loaded_models):
            result[str(model_id)] = {
                "config_path": str(model.config_path),
                "model_path": str(model.model_path),
                "device": model.device,
                "speakers": model.spk2id,
                "styles": list(model.style2id.keys()),
            }
        return result
    
    def get_model_names(self) -> List[str]:
        return [info.name for info in self.model_holder.models_info]
    
    def get_speakers(self, model_id: int = 0) -> Dict[str, int]:
        if model_id >= len(self.loaded_models):
            raise ValueError(f"Model ID {model_id} not found")
        return self.loaded_models[model_id].spk2id
    
    def get_styles(self, model_id: int = 0) -> List[str]:
        if model_id >= len(self.loaded_models):
            raise ValueError(f"Model ID {model_id} not found")
        return list(self.loaded_models[model_id].style2id.keys())


def create_synthesizer(model_dir: Optional[str] = None, device: str = "auto") -> TTSSynthesizer:
    return TTSSynthesizer(model_dir=model_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output file path")
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--speaker", "-s", type=str, help="Speaker name")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--dir", "-d", type=str, help="Model directory")
    
    args = parser.parse_args()
    
    try:
        synthesizer = create_synthesizer(model_dir=args.dir, device=args.device)
        
        print("Available models:", synthesizer.get_model_names())
        print("Available speakers:", list(synthesizer.get_speakers().keys()))
        
        sr, audio = synthesizer.synthesize(
            text=args.text,
            model_name=args.model,
            speaker_name=args.speaker
        )
        
        synthesizer.save_audio(sr, audio, args.output)
        print(f"Audio saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)