import torch
import torch.nn as nn
import torch.nn.functional as F

from shortening import downsample, upsample


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))
        else:
            raise NotImplementedError

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output


class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior, bp_type, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.bp_type = bp_type
        self.threshold = threshold

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )

            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds, gt):
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())
        elif self.bp_type in ['gumbel']:
            assert gt is None
            binomial = torch.distributions.binomial.Binomial(
                preds.size(-1),
                probs=torch.Tensor([self.prior]).to(preds.device)
            )
            loss_boundaries = -binomial.log_prob(
                preds.sum(dim=-1)
            ).mean() / preds.size(-1)

            return loss_boundaries

    def calc_stats(self, preds, gt):
        # B x T
        preds, gt = preds.bool(), gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()

        acc = (preds == gt).sum().item() / gt.numel()

        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        stats = {
            'acc': acc,
            'precision': precision,
            'recall': recall
        }

        return stats


class MemTransformerLM(nn.Module):
    # <CHANGE>
    def __init__(self, n_token, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, pre_lnorm, model_config,
                 activation_function, boundaries_type, spikes_left,
                 temp, prior, add_eot, autoregressive
                 ):
        super(MemTransformerLM, self).__init__()
        self.add_eot = add_eot
        # </CHANGE>
        self.n_token = n_token

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.autoregressive = autoregressive
        
        # <CHANGE>
        n = n_token + 1 if self.add_eot else n_token
        self.word_emb = nn.Embedding(n, d_model)
        # </CHANGE>
        
        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )
        self.r_r_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )

        assert pre_lnorm is False, "We didn't use pre_lnorm"

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm,
                    activation_function=activation_function)
                for _ in range(n_layers)
            ])

            return layers

        pre_layers, (shortened_layers, ), post_layers = eval(model_config)

        self.boundaries_type = boundaries_type
        self.is_bp = boundaries_type in ['unigram', 'entropy', 'gumbel']

        if post_layers == 0 and shortened_layers == 0:
            assert boundaries_type == 'none'
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers)
            ])
        else:
            self.null_group = nn.Parameter(torch.Tensor(1, 1, d_model).zero_())
            nn.init.normal_(self.null_group)

            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers),
                create_decoder_layers(shortened_layers),
                create_decoder_layers(post_layers),
            ])

            self.down_ln = nn.LayerNorm(d_model)

            # Boundary predictor
            if self.is_bp:
                self.boundary_predictor = BoundaryPredictor(
                    d_model=d_model,
                    d_inner=d_inner,
                    activation_function=activation_function,
                    temp=temp,
                    prior=prior,
                    bp_type=boundaries_type,
                )
                self.spikes_left = spikes_left
        # <CHANGE>
        self.final_cast = nn.Linear(d_model, n)
        # </CHANGE>
        self.crit = torch.nn.CrossEntropyLoss(reduction='none')
        self.ar_loss = torch.nn.CosineEmbeddingLoss(reduction='none')

    def _forward(self, core_input, layers):
        # Core_input is of size (T x B x C)
        qlen, _, _ = core_input.size()

        dec_attn_mask = torch.triu(
            core_input.new_ones(qlen, qlen), diagonal=1).bool()

        pos_seq = torch.arange(
            qlen - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype
        )

        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input
        for i, layer in enumerate(layers):
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask
            )

        return core_out

    def get_spikes(self, vector):
        total = torch.ones_like(vector).bool()

        for i in range(1, self.spikes_left + 1, 1):
            mask = vector[i:] > vector[:-i]
            total[i:] &= mask

        return total

    def forward(self,
                data,
                target,
                boundaries_gt,
                replace_threshold = 0):
        """
            data: T x B
            target: T x B
            boundaries_gt: T x B or None
        """
        #<ADDITION>
        B = data.shape[1]
        pad_mask = None
        #</ADDITION>
        stats = {}

        # All batches should be of the same length, but last can be shorter
        tgt_len = target.size(0) if target is not None else data.size(0)

        # Token_ids to vector embeddings -> T x B x C
        word_emb = self.word_emb(data)
        hidden = self.drop(word_emb)
        
        #<ADDITION>
        if self.add_eot:
            eot_emb = self.word_emb(torch.tensor([[self.n_token]], dtype=data.dtype, device=data.device))
        #</ADDITION>
        
        # Extra variables
        loss_boundaries = torch.tensor(0, dtype=data.dtype, device=data.device)
        residual = None
        
        #<ADDITION>
        if self.add_eot:
            residual = word_emb
        #</ADDITION>

        # Process input with Transformer blocks
        for i in range(len(self.layers)):
            if i == 1:  # Downsampling
                #<CHANGE>
                if not self.add_eot:
                    residual = hidden
                #</CHANGE>
                if self.boundaries_type in ['fixed', 'whitespaces']:
                    # T x B
                    hard_boundaries = boundaries_gt.float().transpose(0, 1)
                    # B x T
                else:
                    soft_boundaries, hard_boundaries = self.boundary_predictor(hidden)
                    # B x T

                hidden = downsample(
                    boundaries=hard_boundaries,
                    hidden=hidden,
                    null_group=self.null_group,
                )

                hidden = self.down_ln(hidden)

                pre_token_model = hidden

                # Shortening stats
                stats['p_ones'] = (hard_boundaries.sum() / hard_boundaries.numel()).item()
                stats['loss_boundaries'] = loss_boundaries.item()
                stats['shortened_length'] = hidden.size(0)
            elif i == 2:  # Upsampling
                #<ADDITION>
                pre_upsample = hidden
                #</ADDITION>
                back_hidden = upsample(
                    boundaries=hard_boundaries,
                    shortened_hidden=hidden,
                )
                
                #<ADDITION>
                if self.add_eot:
                    # max number of eot characters to add
                    b_cs = torch.cumsum(hard_boundaries, dim=1).long()
                    num_eots = b_cs.max().item()

                    # get new idx for original token vectors
                    token_ix = torch.arange(back_hidden.shape[0]).unsqueeze(0).to(b_cs.device) + b_cs
                    last_ix = token_ix.max(dim=1).values
                    token_ix = [torch.ones_like(token_ix) * torch.arange(token_ix.shape[0]).unsqueeze(-1).to(token_ix.device), token_ix]
                    token_ix[0] = token_ix[0].reshape(-1)
                    token_ix[1] = token_ix[1].reshape(-1)
                    
                    # put everything together in right place
                    tmp = torch.cat([torch.zeros_like(back_hidden), torch.zeros((num_eots, back_hidden.shape[1], back_hidden.shape[2]), dtype=back_hidden.dtype).to(back_hidden.device)], dim=0)
                    tmp[token_ix[1], token_ix[0], :] = back_hidden.transpose(0,1).reshape(-1, back_hidden.shape[-1])
                    rows, cols = torch.where(tmp.sum(-1) == 0)
                    tmp[rows, cols] = tmp[rows-1, cols]
                    tmp[rows, cols] += eot_emb.squeeze()
                    tmp[token_ix[1], token_ix[0], :] += residual.transpose(0, 1).reshape(-1, residual.shape[-1])
                    hidden = tmp

                    # similarly insert eot marker in target
                    if target is not None:
                        diff = None
                        if target.shape[0] != hard_boundaries.shape[1]:
                            diff = hard_boundaries.shape[1] - target.shape[0]
                            ext = -100 * torch.ones((diff, target.shape[1])).long().to(target.device)
                            target = torch.cat([ext, target], dim=0)
                            tmp = hard_boundaries.clone()
                            tmp[:, :diff] = 0.0
                            b_cs = tmp.cumsum(dim=1).long()
                            num_eots = b_cs.max().item()
                            
                        # need to offset eot target
                        b_cs = torch.cat([b_cs[:, 1:], b_cs[:, -1].unsqueeze(1)], dim=1)
                        token_ix = torch.arange(target.shape[0]).unsqueeze(0).to(b_cs.device) + b_cs
                        last_ix = token_ix.max(dim=1).values
                        token_ix = [torch.ones_like(token_ix) * torch.arange(token_ix.shape[0]).unsqueeze(-1).to(token_ix.device), token_ix]
                        token_ix[0] = token_ix[0].reshape(-1)
                        token_ix[1] = token_ix[1].reshape(-1)
                            
                        tmp = torch.cat([torch.zeros_like(target), torch.zeros((num_eots, target.shape[1])).long().to(target.device)], dim=0)
                        tmp[last_ix, torch.arange(tmp.shape[1])] = 1
                        pad_mask = tmp.cumsum(dim=0)
                        pad_mask[last_ix, torch.arange(tmp.shape[1])] = 0
                        tmp[:] = self.n_token # eot marker
                        tmp[token_ix[1], token_ix[0]] = target.transpose(0,1).reshape(-1)
                        tmp[torch.where(pad_mask == 1)] = -100 # -100 is padding

                        target = tmp
                        
                        if diff is not None:
                            target = target[diff:]
                            pad_mask = pad_mask[diff:]
                    #</ADDITION>

                #<CHANGE>
                else:
                    hidden = back_hidden + residual
                #</CHANGE>
                
            # Out of downsample / upsample -> regular Transformer blocks
            layers = self.layers[i]

            #<ADDITION>
            last_hidden = hidden
            #</ADDITION>
            hidden = self._forward(
                core_input=hidden,
                layers=layers,
            )

            if self.autoregressive and i == 1:
                seq_len, bs, _ = hidden.shape
                keep = (torch.rand((seq_len-1, bs, 1), device = hidden.device).detach() > replace_threshold).to(hidden.dtype)
                temp = keep*hidden[:-1] + (1-keep)*pre_token_model[1:]
                hidden = torch.cat((temp, hidden[-1].unsqueeze(0)))

        # Calculate loss
        #<CHANGE>
        hidden = hidden[-target.size(0):] if target is not None else hidden[-tgt_len:]
        #</CHANGE>
        logit = self.final_cast(hidden)

        if self.training or target is not None:
            # T x B x C
            assert hidden.size(0) == target.size(0)

            # Boundary predictor loss
            if self.is_bp:
                if self.boundaries_type == 'entropy':
                    entropy = -torch.nn.functional.log_softmax(
                        logit, dim=-1
                    ) * torch.nn.functional.softmax(logit, dim=-1)

                    entropy = torch.sum(entropy, dim=-1)
                    # T x B

                    target_boundaries = self.get_spikes(entropy).transpose(0, 1)
                    # target_boundaries: B x T
                elif self.boundaries_type == 'unigram':
                    # T x B
                    target_boundaries = boundaries_gt[-tgt_len:].transpose(0, 1)
                    # B x T
                elif self.boundaries_type == 'gumbel':
                    target_boundaries = None

                soft_boundaries = soft_boundaries[:, -tgt_len:]
                hard_boundaries = hard_boundaries[:, -tgt_len:]

                if self.boundaries_type in ['unigram', 'entropy']:
                    assert target_boundaries.sum().item() > 0

                    loss_boundaries = self.boundary_predictor.calc_loss(
                        soft_boundaries, target_boundaries
                    )

                    bp_stats = self.boundary_predictor.calc_stats(
                        hard_boundaries, target_boundaries
                    )

                    for k, v in bp_stats.items():
                        stats[f'{k}'] = v
                elif self.boundaries_type == 'gumbel':
                    loss_boundaries = self.boundary_predictor.calc_loss(
                        preds=hard_boundaries, gt=None
                    )

                    bp_stats = self.boundary_predictor.calc_stats(
                        hard_boundaries, (data == 0)[-tgt_len:].transpose(0, 1)
                    )

                    for k, v in bp_stats.items():
                        stats[f'{k}'] = v

                stats['loss_boundaries'] = loss_boundaries.item()

            # LM loss
            #<CHANGE>
            if pad_mask is not None:
                ix = torch.where(pad_mask != 1)
                logit = logit[ix]
                target = target[ix]
            else:
                logit = logit.view(-1, logit.size(-1))
                target = target.view(-1)

            loss = self.crit(logit, target)
            if self.autoregressive:
                loss_token = self.ar_loss(pre_token_model[1:].view(-1, pre_upsample.size(-1)), pre_upsample[:-1].view(-1, pre_upsample.size(-1)), torch.ones((pre_upsample.size(0)-1)*pre_upsample.size(1), dtype = pre_upsample.dtype, device = pre_upsample.device).detach())
            else:
                loss_token = torch.tensor(0., device = pre_upsample.device, dtype = pre_upsample.dtype, requires_grad = True)
            
            return loss, stats, loss_boundaries, logit, hard_boundaries[:, -tgt_len:], loss_token
        else:
            # Generation mode, we return raw logits
            return logit, hard_boundaries, pre_upsample, last_hidden
            #</CHANGE>

    #<ADDITION>
    def generate_next(self, data, boundaries_gt = None, num_tokens=1):
        """Generates tokens."""
        count = -1 if self.add_eot else 0
        out = []
        orig_len = data.shape[0]
        with torch.no_grad():
            while count != num_tokens:
                if len(out) == 0:
                    logit, hard_boundaries, pre_upsample, last_hidden = self.forward(data, None, boundaries_gt)
                    logit = logit.squeeze()
                    out = [logit[-1].argmax().item()]

                # finished token, start a new one
                if out[-1] == self.n_token:
                    count += 1

                    if count == num_tokens:
                        # data = torch.cat([data, torch.LongTensor(out[:-1]).to(data.device).unsqueeze(-1)], dim=0)
                        return data[orig_len:].squeeze()

                    # sample first character
                    char_vec = self.word_emb(torch.tensor([[out[-1]]], dtype=data.dtype, device=data.device))
                    new_vec = pre_upsample[-1].unsqueeze(0) + char_vec
                    last_hidden = torch.cat([last_hidden, new_vec], dim=0)
                    hidden = self._forward(core_input=last_hidden, layers = self.layers[-1])
                    logit = self.final_cast(hidden).squeeze()
                    out = [logit[-1].argmax().item()]

                    # reset pre_upsample
                    data = torch.cat([data, torch.LongTensor([out]).to(data.device)], dim=0)
                    logit, hard_boundaries, pre_upsample, last_hidden = self.forward(data, None, boundaries_gt)
                    logit = logit.squeeze()
                    out = [logit[-1].argmax().item()]

                # continue rest of token
                while out[-1] != self.n_token:
                    char_vec = self.word_emb(torch.tensor([[out[-1]]], dtype=data.dtype, device=data.device))
                    new_vec = pre_upsample[-1].unsqueeze(0) + char_vec
                    last_hidden = torch.cat([last_hidden, new_vec], dim=0)
                    hidden = self._forward(core_input=last_hidden, layers = self.layers[-1])
                    logit = self.final_cast(hidden).squeeze()
                    out.append(logit[-1].argmax().item())
                data = torch.cat([data, torch.LongTensor(out[:-1]).to(data.device).unsqueeze(-1)], dim=0)