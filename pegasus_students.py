import warnings
import torch
from torch import nn
from typing import Optional, Tuple, List, Union
from transformers import PegasusModel, PegasusConfig, PegasusForConditionalGeneration

#Student configuration

import warnings
import torch
from torch import nn
from typing import Optional, Tuple, List, Union
from transformers import PegasusModel, PegasusConfig, PegasusForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers import SummarizationPipeline

students_config_book = {
    '2': PegasusConfig(encoder_layers=2, decoder_layers=2),
    '4': PegasusConfig(encoder_layers=4, decoder_layers=4),
    '6': PegasusConfig(encoder_layers=6, decoder_layers=6),
    '8': PegasusConfig(encoder_layers=8, decoder_layers=8),
    '10': PegasusConfig(encoder_layers=10, decoder_layers=10),
    '12': PegasusConfig(encoder_layers=12, decoder_layers=12),
    '16': PegasusConfig(encoder_layers=16, decoder_layers=16)
}


LAYERS_TO_COPY = {
    16: {  # maps  num layers in student -> which teacher layers to copy
        1: [0],
        2: [0, 15],
        3: [0, 8, 15],
        4: [0, 5, 10, 15],
        6: [0, 3, 6, 9, 12, 15],
        8: [0, 2, 4, 6, 8, 10, 12, 15],
        9: [0, 1, 3, 5, 7, 9, 11, 13, 15],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        16: list(range(16)),
    },
}
LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
}


def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy) -> None:
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(
        layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())

# Copied from transformers.models.bart.modeling_bart.shift_tokens_right


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))


def get_layers_to_supervise(n_student, n_teacher) -> List[int]:
    """Used or the --supervise_forward kwarg"""
    if n_student > n_teacher:
        raise ValueError(
            f"Cannot perform intermediate supervision for student {n_student} > teacher {n_teacher}")
    elif n_teacher == n_student:
        return list(range(n_teacher))
    elif n_student == 1:
        return [n_teacher - 1]
    else:
        return LAYERS_TO_SUPERVISE[n_teacher][n_student]


def create_student_with_configuration(teacher,
                                      e=None,
                                      d=None,
                                      copy_first_teacher_layers = False,
                                      save_path='./student'):

    teacher.eval()
    teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
    init_kwargs = teacher.config.to_diff_dict()
    if e is None:
        e = teacher_e
    if d is None:
        d = teacher_d
    init_kwargs.update({"encoder_layers": e, "decoder_layers": d})
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForSeq2SeqLM.from_config(student_cfg)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    # every student key should have a teacher keys.
    assert info.missing_keys == [], info.missing_keys

    if copy_first_teacher_layers:  # Our copying is done. We just log and save
        e_layers_to_copy, d_layers_to_copy = list(range(e)), list(range(d))
        student.save_pretrained(save_path)
        return student, e_layers_to_copy, d_layers_to_copy

    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    e_layers_to_copy: List[int] = pick_layers_to_copy(e, teacher_e)
    d_layers_to_copy: List[int] = pick_layers_to_copy(d, teacher_d)

    copy_layers(teacher.model.encoder.layers,
                student.model.encoder.layers, e_layers_to_copy)
    copy_layers(teacher.model.decoder.layers,
                student.model.decoder.layers, d_layers_to_copy)

    student.config.init_metadata = dict(
        teacher_type=teacher.config.model_type,
        copied_encoder_layers=e_layers_to_copy,
        copied_decoder_layers=d_layers_to_copy,
    )
    student.save_pretrained(save_path)
    # Save information about copying for easier reproducibility

    return student

