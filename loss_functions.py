
import logging
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration,PegasusConfig,AutoTokenizer,AutoModelForSeq2SeqLM

import torch
from torch.nn import functional as F
from torch import nn
from typing import Callable, Dict, Iterable, List, Tuple, Union

import gc

import numpy as np


def cross_entropy_loss(logits, labels,label_smoothing,pad_token_id):
    lprobs = F.log_softmax(logits, dim=-1)
    student_lm_loss, _ = label_smoothed_nll_loss(
                lprobs, labels, label_smoothing, ignore_index=pad_token_id
            )
    return student_lm_loss

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def calc_hidden_loss(attention_mask, hidden_states, hidden_states_T, matches, normalize_hidden):
    """MSE(student_hid, teacher_hid[matches]). Called "Intermediate supervision" in paper. Inspired by TinyBERT."""
    msg = "expected list or tuple for hidden_states, got tensor of shape: "
    mask = attention_mask.to(hidden_states[0])
    valid_count = mask.sum() * hidden_states[0].size(-1)
    student_states = torch.stack([hidden_states[i] for i in range(len(matches))])
    teacher_states = torch.stack([hidden_states_T[j] for j in matches])
    if normalize_hidden:
        student_states = F.layer_norm(student_states, student_states.shape[1:])
        teacher_states = F.layer_norm(teacher_states, teacher_states.shape[1:])
    mse = F.mse_loss(student_states, teacher_states, reduction="none")
    masked_mse = (mse * mask.unsqueeze(0).unsqueeze(-1)).sum() / valid_count
    return masked_mse


def calc_ce_loss(mask, s_logits, t_logits,temperature=2):
    """Copy pasted from distillbert (transformers/examples/distillation/)"""
    # mask has False at padding_idx
    sel_mask = mask[:, :, None].expand_as(s_logits)

    #print(sel_mask.shape ,s_logits.shape ,t_logits.shape )
    
    vocab_size = s_logits.size(-1)
    s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
    t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
    ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
    loss_ce = (
        ce_loss_fct(
            F.log_softmax(s_logits_slct / temperature, dim=-1),
            F.softmax(t_logits_slct / temperature, dim=-1),
        )
        * (temperature) ** 2 
    )
    return loss_ce

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    #print(pad_token_id, input_ids)
    x= (input_ids.ne(pad_token_id).sum(dim=1) - 1)
    index_of_eos = x.unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def blended_loss(teacher,student,batch,labels,pad_token_id): #TODO Calculate the total loss
        
        alpha_hid = 0.2
        alpha_ce = 0.8
        alpha_mlm= 0.8
        
        e_matches = list(range(student.config.encoder_layers))
        
        d_matches = list(range(student.config.decoder_layers))
        
        
        different_base_models = not (student is None or teacher == student)
        do_calc_hidden_loss = (not different_base_models) and alpha_hid > 0
        different_encoder = different_base_models or (student.config.encoder_layers != teacher.config.encoder_layers)
        
        input_ids,src_mask = batch['input_ids'],batch['attention_mask']
        decoder_input_ids = shift_tokens_right(labels, pad_token_id)
        
        student_outputs = student(input_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False)
        lm_logits = student_outputs["logits"]
        label_smoothing = 0.1
        student_loss = cross_entropy_loss(lm_logits, labels,label_smoothing,pad_token_id)
        def zero_tensor():
            return torch.tensor(0.0).type_as(student_loss)
        teacher_enc_outputs = student_outputs[
            "encoder_last_hidden_state"
        ]  # use this unless self.different_base_models
        hid_loss_enc, hid_loss_dec = zero_tensor(), zero_tensor()
        
        if different_encoder:  # compute encoder hidden state loss
            all_teacher_encoder_outputs = teacher.get_encoder()(
                input_ids,
                attention_mask=src_mask,
                output_hidden_states=True,
            )
            if different_base_models:
                teacher_enc_outputs = all_teacher_encoder_outputs["last_hidden_state"]
            elif do_calc_hidden_loss:
                hid_loss_enc = calc_hidden_loss(
                    src_mask,
                    student_outputs["encoder_hidden_states"],
                    all_teacher_encoder_outputs["hidden_states"],
                    e_matches,
                    normalize_hidden=True,
                )
        # decoder_input_ids for teacher [8,1] (zeros)
        #decodeIds = torch.cuda.LongTensor([0,0,0,0,0,0,0,0]).reshape(1,8)
        teacher_outputs = teacher(
            input_ids,
            attention_mask=src_mask,
            encoder_outputs=(teacher_enc_outputs,),
            decoder_input_ids= decoder_input_ids,
            output_hidden_states=do_calc_hidden_loss,
            use_cache=False,  # since we are not passing labels, never let this default to True
        )
        dec_mask = decoder_input_ids.ne(pad_token_id)
        loss_ce = calc_ce_loss(dec_mask, lm_logits, teacher_outputs["logits"])
        if do_calc_hidden_loss:  # Intermediate supervision of decoder hidden states
            hid_loss_dec = calc_hidden_loss(
                dec_mask,
                student_outputs["decoder_hidden_states"],
                teacher_outputs["decoder_hidden_states"],
                d_matches,
                normalize_hidden=True,
            )

        blended_loss = (
            alpha_ce * loss_ce
            + alpha_mlm * student_loss
            + alpha_hid * (hid_loss_enc + hid_loss_dec)
        )
        return blended_loss

