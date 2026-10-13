"""Check model-family dispatch without allocating weights or accessing GPUs."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize('kind,path,expected', [
    ('qwen3_5', '/checkpoints/aurav2', 'Qwen3_5ForConditionalGeneration'),
    ('qwen3_vl', '/checkpoints/aura', 'Qwen3VLForConditionalGeneration'),
    ('qwen3_vl', '/checkpoints/Qwen-A3B', 'Qwen3VLMoeForConditionalGeneration'),
])
def test_dispatch_loads_only_selected_architecture(kind, path, expected):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('loading_server', root / 'hf_openai_server.py')
    server = importlib.util.module_from_spec(spec)
    classes = {name: MagicMock() for name in (
        'AutoProcessor', 'Qwen3_5ForConditionalGeneration',
        'Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration',
    )}
    with patch.dict(sys.modules, {
        'torch': SimpleNamespace(),
        'transformers': SimpleNamespace(**classes),
        'qwen_vl_utils': SimpleNamespace(process_vision_info=MagicMock()),
    }):
        spec.loader.exec_module(server)
        with patch.object(server, 'resolve_torch_dtype', return_value='bf16'):
            engine = server.build_engine(
                model_type=kind, model_path=path, device_map='auto',
                torch_dtype='bf16', attn_implementation='flash_attention_2',
                use_audio_in_video=False,
            )
    classes[expected].from_pretrained.assert_called_once_with(
        path, torch_dtype='bf16', device_map='auto',
        attn_implementation='flash_attention_2', trust_remote_code=True,
    )
    for name, model in classes.items():
        if name not in ('AutoProcessor', expected):
            model.from_pretrained.assert_not_called()
    assert engine.model_path == path


def test_qwen35_preserves_per_video_metadata_without_fps_list():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('metadata_server', root / 'hf_openai_server.py')
    server = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': SimpleNamespace()}):
        spec.loader.exec_module(server)
    engine = server.Qwen3_5Engine.__new__(server.Qwen3_5Engine)
    engine.processor = MagicMock()
    engine.processor.image_processor.patch_size = 16
    engine.model = SimpleNamespace(device='cpu')
    metadata = [
        {'fps': 2.0, 'frames_indices': [0, 1], 'total_num_frames': 2},
        {'fps': 4.0, 'frames_indices': [2, 4], 'total_num_frames': 6},
    ]
    videos = [object(), object()]
    engine.process_vision_info = MagicMock(return_value=(
        None, list(zip(videos, metadata)), {'do_sample_frames': False},
    ))
    engine._prepare_multimodal_inputs([{'role': 'user'}], 'prompt')
    engine.process_vision_info.assert_called_once_with(
        [{'role': 'user'}], return_video_kwargs=True,
        return_video_metadata=True, image_patch_size=16,
    )
    engine.processor.assert_called_once_with(
        text=['prompt'], images=None, videos=videos, video_metadata=metadata,
        return_tensors='pt', videos_kwargs={'do_sample_frames': False},
    )
    engine.processor.return_value.to.assert_called_once_with('cpu')


@pytest.mark.parametrize('family,configured,expected', [
    ('Qwen3_5Engine', 248044, [248044, 248046]),
    ('Qwen3_5Engine', [248044, 248099], [248044, 248099, 248046]),
    ('Qwen3_5Engine', [248044, 248046], [248044, 248046]),
    ('Qwen3VLEngine', 151645, None),
])
def test_generation_stops_at_turn_end_only_for_qwen35(family, configured, expected):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('stop_server', root / 'hf_openai_server.py')
    server = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': SimpleNamespace(no_grad=nullcontext)}):
        spec.loader.exec_module(server)
    engine = getattr(server, family).__new__(getattr(server, family))
    engine.model = MagicMock()
    engine.model.generation_config.eos_token_id = configured
    tokenizer = SimpleNamespace(
        eos_token_id=248046,
        convert_tokens_to_ids=lambda token: 248046,
        convert_ids_to_tokens=lambda token: '<|im_end|>',
    )
    engine.processor = MagicMock()
    engine.processor.tokenizer = tokenizer
    # Keep the native silence token visible during decoding.
    engine.processor.batch_decode.return_value = ['<|silent|><|im_end|>']
    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 4))
    engine._prepare_multimodal_inputs = MagicMock(return_value=Inputs())
    result = engine.generate([], max_new_tokens=1024, temperature=0.0, skip_special_tokens=False)
    kwargs = engine.model.generate.call_args.kwargs
    if expected is None:
        assert 'eos_token_id' not in kwargs
    else:
        assert kwargs['eos_token_id'] == expected
    assert engine.model.generation_config.eos_token_id == configured
    assert kwargs['max_new_tokens'] == 1024
    assert kwargs['do_sample'] is False
    assert result == '<|silent|><|im_end|>'
    assert engine.processor.batch_decode.call_args.kwargs['skip_special_tokens'] is False
