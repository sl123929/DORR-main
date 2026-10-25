import numpy as np
import torch


# def get_emb(state_tracker, buffer, indices=None, is_obs=None, obs=None, remove_recommended_ids=False):
#     if len(buffer) == 0:
#         obs_emb, recommended_ids = state_tracker.forward(obs=obs, reset=True,
#                                                          remove_recommended_ids=remove_recommended_ids)
#     else:
#         if indices is None:  # collector collects data
#             indices = buffer.last_index[~buffer[buffer.last_index].done]
#             is_obs = False
#
#         obs_emb, recommended_ids = state_tracker.forward(buffer=buffer, indices=indices, reset=False, is_obs=is_obs,
#                                                          remove_recommended_ids=remove_recommended_ids)
#
#     if recommended_ids is not None:
#         recommended_ids = recommended_ids[:, :-1]
#     return obs_emb, recommended_ids

def get_emb(state_tracker, buffer, indices=None, is_obs=None, obs=None, is_train=True):
    if len(buffer) == 0:
        obs_emb = state_tracker.forward(obs=obs, reset=True)
    else:
        if indices is None:  # collector collects data
            indices = buffer.last_index[~buffer[buffer.last_index].done]
            ## ~是取反，后面如果是True则变为False，所以indices会取出last_index中done为false的index
            is_obs = False

        obs_emb = state_tracker.forward(buffer=buffer, indices=indices, reset=False, is_obs=is_obs, is_train=is_train)

    # print("======================================================")
    # print(obs_emb.shape)
    # print("======================================================")

    return obs_emb

def removed_recommended_id_from_embedding(logits, recommended_ids, return_valid_mask=False):
    """
    :param logits: Batch * Num_all_items
    :param recommended_ids: Batch * Num_removed
    :return:
    :rtype:
    """

    num_batch, num_action = logits.shape

    indices = np.expand_dims(np.arange(num_action), 0).repeat(num_batch, axis=0) # 得到一个batch size*num_action的tensor
    indices_torch = torch.from_numpy(indices).to(device=logits.device, dtype=torch.long)

    if recommended_ids is None:
        if return_valid_mask:
            return logits, indices_torch, torch.ones_like(logits, dtype=torch.bool)
        return logits, indices_torch

    # assert all(recommended_ids[:, -1] == num_action)
    # recommended_ids_valid = recommended_ids[:, :-1]
    # recommended_ids_valid_torch = torch.LongTensor(recommended_ids_valid).to(device=logits.device)
    recommended_ids_valid_torch = torch.LongTensor(recommended_ids).to(device=logits.device)

    mask = torch.ones_like(logits, dtype=torch.bool)
    mask_valid = mask.scatter(1, recommended_ids_valid_torch, 0)

    counts = mask_valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("No unrecommended actions remain for at least one user")
    # Different numbers of unique historical items imply ragged candidate rows.
    # Compact each row separately; pad with a valid ID and zero probability.
    width = int(counts.max().item())
    ordered = indices_torch.masked_fill(~mask_valid, num_action).sort(dim=1).values[:, :width]
    valid_slots = torch.arange(width, device=logits.device)[None, :] < counts[:, None]
    indices_masked = torch.where(valid_slots, ordered, ordered[:, :1])
    logits_masked = logits.gather(1, indices_masked).masked_fill(~valid_slots, 0)
    # Categorical receives probabilities, not log scores. If all remaining
    # mass underflows to zero, use uniform mass on real candidates only.
    zero_mass = logits_masked.sum(1) <= 0
    logits_masked = torch.where(zero_mass[:, None], valid_slots.to(logits.dtype), logits_masked)
    if return_valid_mask:
        return logits_masked, indices_masked, valid_slots
    return logits_masked, indices_masked


def get_recommended_ids(buffer):
    if len(buffer) == 0:
        recommended_ids = None
    else:
        indices = buffer.last_index[~buffer[buffer.last_index].done]

        # is_alive = True
        recommended_ids = np.zeros([0, len(indices)], dtype=int)
        while True:
            acts = buffer.act[indices]
            recommended_ids = np.vstack([recommended_ids, acts])

            if all(indices == buffer.prev(indices)):
                break
            assert all(indices != buffer.prev(indices))

            indices = buffer.prev(indices)

        recommended_ids = recommended_ids.T

    return recommended_ids