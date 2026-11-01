import pytest
import torch

from evidence.safetensors_header import TensorEntry
from evidence.views import plan_views, decode_view, component_of

def E(**s): return {n: TensorEntry(n, d, sh, "f.safetensors", 0, 0) for n, (d, sh) in s.items()}

def test_fused_experts_slice_and_orient():
    cfg = {"hidden_size": 2048, "num_experts": 128}
    v, _ = plan_views(E(**{"layers.0.mlp.experts.gate_up_proj": ("BF16", (128, 2048, 1536))}), cfg, all_experts=True)
    gate = [x for x in v if x.name.startswith("layers.0.mlp.experts.gate_proj[")]
    assert len(gate) == 128 and gate[0].matrix_shape == (768, 2048)       # split gate/up, oriented (out,in)

def test_expert_subset_default():
    cfg = {"hidden_size": 2048, "num_experts": 128}
    v, _ = plan_views(E(**{"layers.0.mlp.experts.down_proj": ("BF16", (128, 768, 2048))}), cfg)
    assert len(v) == 8

def test_unfused_expert_subset():
    cfg = {"hidden_size": 64, "num_experts": 16}
    v, cov = plan_views(E(**{f"layers.0.mlp.experts.{i}.down_proj.weight": ("BF16", (64, 32)) for i in range(16)}), cfg)
    assert len(v) == 8 and [c["reason"] for c in cov].count("expert_not_sampled") == 8

def test_conv1d_and_whisper_not_sliced():
    v, cov = plan_views(E(**{"encoder.conv1.weight": ("F32", (768, 80, 3)),
                             "layers.0.linear_attn.conv1d.weight": ("BF16", (8192, 1, 4))}), {})
    assert [x.geometry for x in v] == ["conv_flat"] and v[0].matrix_shape == (768, 240)
    assert {c["name"]: c["reason"] for c in cov}["layers.0.linear_attn.conv1d.weight"] == "vector_like"

def test_scale_tensors_never_matrices():
    v, cov = plan_views(E(**{"m.q.weight": ("U8", (512, 1024)), "m.q.weight_scale": ("F8_E4M3", (512, 128)),
                             "m.q.weight_scale_2": ("F32", ()), "g.k.qweight": ("I32", (8, 64)),
                             "g.k.scales": ("F16", (38, 896)), "g.k.qzeros": ("I32", (38, 112))}), {})
    assert [x.decode for x in v] == ["nvfp4"]
    reasons = {c["name"]: c["reason"] for c in cov}
    assert reasons["g.k.scales"] == "unsupported_representation"     # module poisoned by qweight

def test_fp8_block_size_from_config_and_unknown():
    entries = E(**{"kv_a.weight": ("F8_E4M3", (576, 7168)), "kv_a.weight_scale_inv": ("F32", (5, 56))})
    v, _ = plan_views(entries, {"quantization_config": {"weight_block_size": [128, 128]}})
    assert v[0].decode == "fp8_scaled"
    _, cov = plan_views(entries, {})
    assert {c["name"]: c["reason"] for c in cov}["kv_a.weight"] == "fp8_block_size_unknown"

def test_lora_families_and_scaling():
    kohya = E(**{"lora_unet_x_attn1_to_k.lora_down.weight": ("BF16", (32, 640)),
                 "lora_unet_x_attn1_to_k.lora_up.weight": ("BF16", (640, 32)),
                 "lora_unet_x_attn1_to_k.alpha": ("BF16", ())})
    v, cov = plan_views(kohya, {})
    assert [x.geometry for x in v] == ["lora_product"] and not [c for c in cov if c["status"] == "skipped"]
    peft = E(**{"base_model.model.q_proj.lora_A.weight": ("F32", (16, 64)),
                "base_model.model.q_proj.lora_B.weight": ("F32", (64, 16))})
    v, _ = plan_views(peft, {}, adapter_config={"r": 16, "lora_alpha": 32})
    assert v[0].scale == 2.0
    v, _ = plan_views(peft, {}, adapter_config={"r": 16, "lora_alpha": 32, "use_rslora": True})
    assert v[0].scale == pytest.approx(32 / 4)

def test_unsupported_adapter_types():
    _, cov = plan_views(E(**{"m.hada_w1_a": ("F32", (8, 64))}), {})
    assert cov[0]["reason"] == "unsupported_adapter_type"

def test_decision_head_kept_and_depthwise_conv_skipped():
    v, cov = plan_views(E(**{"classifier.weight": ("F32", (2, 384)),
                             "layers.0.conv1d.weight": ("BF16", (8192, 1, 4))}), {})
    assert [(x.name, x.geometry) for x in v] == [("classifier.weight", "head")]
    assert {c["name"]: c["reason"] for c in cov}["layers.0.conv1d.weight"] == "vector_like"

def test_integer_buffers_do_not_poison():
    v, cov = plan_views(E(**{"attn.qkv.weight": ("F16", (432, 144)),
                             "attn.relative_position_index": ("I64", (144, 144)),
                             "decoder.upsample_factor": ("I64", ()),
                             "decoder.upsample.weight": ("F32", (64, 64, 2))}), {})
    assert {x.name for x in v} == {"attn.qkv.weight", "decoder.upsample.weight"}
    assert {c["name"]: c["reason"] for c in cov}["attn.relative_position_index"] == "non_float_buffer"

def test_mlx_and_embedded_peft_lora():
    mlx = E(**{"layers.0.w1.lora_a": ("F32", (2048, 16)), "layers.0.w1.lora_b": ("F32", (16, 8192))})
    v, _ = plan_views(mlx, {}, adapter_config={"lora_parameters": {"rank": 16, "scale": 2.0}})
    assert v[0].geometry == "lora_product" and v[0].matrix_shape == (8192, 2048) and v[0].scale == 2.0
    embedded = E(**{"model.layers.0.q_proj.weight": ("BF16", (64, 64)),
                    "base_model.model.model.layers.0.q_proj.lora_A.weight": ("BF16", (8, 64)),
                    "base_model.model.model.layers.0.q_proj.lora_B.weight": ("BF16", (64, 8))})
    v, cov = plan_views(embedded, {})
    assert {x.geometry for x in v} == {"matrix", "lora_product"}
    assert "lora_scale_unknown" in {c["reason"] for c in cov}

def test_deepseek_fp4_experts_and_e8m0_fp8():
    cfg = {"hidden_size": 5120, "num_experts": 256,
           "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32],
                                   "scale_fmt": "ue8m0", "expert_dtype": "fp4"}}
    v, _ = plan_views(E(**{"layers.1.ffn.experts.0.w1.weight": ("I8", (2304, 2560)),
                           "layers.1.ffn.experts.0.w1.scale": ("F8_E8M0", (2304, 160)),
                           "layers.1.attn.wkv.weight": ("F8_E4M3", (512, 5120)),
                           "layers.1.attn.wkv.scale": ("F8_E8M0", (16, 160))}), cfg, all_experts=True)
    by = {x.name: x for x in v}
    assert by["layers.1.ffn.experts.0.w1.weight"].decode == "mxfp4"
    assert by["layers.1.ffn.experts.0.w1.weight"].matrix_shape == (2304, 5120)
    assert by["layers.1.attn.wkv.weight"].decode == "fp8_scaled"


def reader(tensors):
    return lambda name, rows: tensors[name] if rows is None else tensors[name][rows]


def test_fp4_values_nibbles_and_block_scales():
    # Verified stored layouts: low nibble first and row-major block scales.
    packed = torch.tensor([[0x21, 0xF7] * 8, [0xA3, 0x64] * 8], dtype=torch.uint8)
    cfg = {"quantization_config": {"expert_dtype": "fp4"}}
    ds = E(**{"m.weight": ("I8", (2, 16)), "m.scale": ("F8_E8M0", (2, 1))})
    v, cov = plan_views(ds, cfg)
    raw = {"m.weight": packed.view(torch.int8), "m.scale": torch.tensor([[128], [126]], dtype=torch.uint8)}
    actual = decode_view(v[0], reader(raw), cfg)
    assert actual.dtype == torch.float32 and actual.shape == (2, 32)
    torch.testing.assert_close(actual[0, :4], torch.tensor([1., 2., 12., -12.]))
    torch.testing.assert_close(actual[1, :4], torch.tensor([.75, -.5, 1., 2.]))
    assert {x["status"] for x in cov} == {"measured", "consumed"}
    nv = E(**{"m.weight": ("U8", (2, 16)), "m.weight_scale": ("F8_E4M3", (2, 2)), "m.weight_scale_2": ("F32", ())})
    v, _ = plan_views(nv, {})
    raw = {"m.weight": packed, "m.weight_scale": torch.tensor([[2., 4.], [1., .5]]).to(torch.float8_e4m3fn), "m.weight_scale_2": torch.tensor(.25)}
    actual = decode_view(v[0], reader(raw), {})
    torch.testing.assert_close(actual[0, :4], torch.tensor([.25, .5, 3., -3.]))
    torch.testing.assert_close(actual[0, 16:20], torch.tensor([.5, 1., 6., -6.]))
    gpt = E(**{"m.experts.down_proj_blocks": ("U8", (2, 2, 1, 16)), "m.experts.down_proj_scales": ("U8", (2, 2, 1))})
    cfg = {"model_type": "gpt_oss", "num_experts": 2}
    v, _ = plan_views(gpt, cfg, all_experts=True)
    raw = {"m.experts.down_proj_blocks": packed.reshape(1, 2, 1, 16).repeat(2, 1, 1, 1), "m.experts.down_proj_scales": torch.tensor([[[128], [126]], [[127], [129]]], dtype=torch.uint8)}
    actual = decode_view(v[1], reader(raw), cfg)
    torch.testing.assert_close(actual[0, :4], torch.tensor([.5, 1., 6., -6.]))
    torch.testing.assert_close(actual[1, :4], torch.tensor([6., -4., 8., 16.]))


@pytest.mark.parametrize("scale,config,expected", [
    (torch.tensor(2.), {}, torch.full((3, 8), 2.)),
    (torch.tensor([1., 2., 4.]), {}, torch.tensor([[1.] * 8, [2.] * 8, [4.] * 8])),
    (torch.tensor([[1., 2.], [3., 4.]]), {"quantization_config": {"weight_block_size": [2, 4]}}, torch.tensor([[1.] * 4 + [2.] * 4] * 2 + [[3.] * 4 + [4.] * 4])),
])
def test_fp8_scale_multiplies_scalar_channel_and_partial_blocks(scale, config, expected):
    entries = E(**{"m.weight": ("F8_E4M3", (3, 8)), "m.weight_scale_inv": ("F32", tuple(scale.shape))})
    v, _ = plan_views(entries, config)
    raw = {"m.weight": torch.ones(3, 8).to(torch.float8_e4m3fn), "m.weight_scale_inv": scale}
    torch.testing.assert_close(decode_view(v[0], reader(raw), config), expected)


def test_fp8_per_expert_e8m0_and_fused_orientation_values():
    cfg = {"hidden_size": 8, "num_experts": 2, "quantization_config": {"weight_block_size": [4, 4], "scale_fmt": "ue8m0"}}
    entries = E(**{"m.experts.gate_up_proj": ("F8_E4M3", (2, 8, 16)), "m.experts.gate_up_proj_scale": ("F8_E8M0", (2, 2, 4))})
    v, _ = plan_views(entries, cfg, all_experts=True)
    raw = {"m.experts.gate_up_proj": torch.arange(256).reshape(2, 8, 16).remainder(8).to(torch.float8_e4m3fn), "m.experts.gate_up_proj_scale": torch.full((2, 2, 4), 128, dtype=torch.uint8)}
    gate = next(x for x in v if x.name == "m.experts.gate_proj[1]")
    torch.testing.assert_close(decode_view(gate, reader(raw), cfg), raw[gate.sources[0]].float()[1].T[:8] * 2)


def test_lora_runtime_alpha_mlx_and_patterns_values():
    entries = E(**{"m.lora_down.weight": ("F32", (2, 8)), "m.lora_up.weight": ("F32", (8, 2)), "m.alpha": ("F32", ())})
    v, _ = plan_views(entries, {})
    raw = {"m.lora_down.weight": torch.ones(2, 8), "m.lora_up.weight": torch.ones(8, 2), "m.alpha": torch.tensor(6.)}
    torch.testing.assert_close(decode_view(v[0], reader(raw), {}), torch.full((8, 8), 6.))
    entries = E(**{"m.lora_a": ("F32", (8, 2)), "m.lora_b": ("F32", (2, 16))})
    v, _ = plan_views(entries, {}, {"lora_parameters": {"scale": 3.}})
    raw = {"m.lora_a": torch.ones(8, 2), "m.lora_b": torch.ones(2, 16)}
    torch.testing.assert_close(decode_view(v[0], reader(raw), {}), torch.full((16, 8), 6.))
    entries = E(**{"base_model.model.m.lora_A.weight": ("F32", (2, 8)), "base_model.model.m.lora_B.weight": ("F32", (8, 2))})
    v, _ = plan_views(entries, {}, {"r": 4, "lora_alpha": 4, "rank_pattern": {"m": 2}, "alpha_pattern": {"m": 8}})
    assert v[0].scale == 4.


def test_plain_heads_convs_and_component_values():
    entries = E(**{"score.weight": ("F32", (1, 8)), "unet.down_blocks.1.conv.weight": ("F32", (2, 4, 3))})
    v, _ = plan_views(entries, {})
    tensors = {"score.weight": torch.arange(8).reshape(1, 8), "unet.down_blocks.1.conv.weight": torch.arange(24).reshape(2, 4, 3)}
    for view in v:
        torch.testing.assert_close(decode_view(view, reader(tensors), {}), tensors[view.name].float().reshape(view.matrix_shape))
    assert component_of("model.visual.blocks.3.attn.qkv.weight", "model.safetensors") == "model.visual.blocks"
    assert component_of("down_blocks.1.conv.weight", "unet/model.safetensors") == "unet/down_blocks"


def test_fp8_nonaligned_row_block_reads_covering_scales():
    config = {"quantization_config": {"weight_block_size": [128, 8]}}
    entries = E(**{"m.weight": ("F8_E4M3", (576, 8)), "m.weight_scale_inv": ("F32", (5, 1))})
    view = plan_views(entries, config)[0][0]
    tensors = {"m.weight": torch.ones(576, 8).to(torch.float8_e4m3fn), "m.weight_scale_inv": torch.tensor([[1.], [2.], [4.], [8.], [16.]])}
    full = decode_view(view, reader(tensors), config)
    requests = []

    def read(name, rows):
        requests.append((name, rows))
        return reader(tensors)(name, rows)

    block = decode_view(view, read, config, row_slice=slice(160, 416))
    torch.testing.assert_close(block, full[160:416])
    assert requests == [("m.weight", slice(160, 416)), ("m.weight_scale_inv", slice(1, 4))]


def test_row_slice_keeps_other_geometry_coordinates():
    entries = E(**{"m.experts.gate_up_proj": ("F32", (2, 8, 16)), "conv.weight": ("F32", (8, 4, 3)), "m.lora_A.weight": ("F32", (2, 8)), "m.lora_B.weight": ("F32", (8, 2)), "score.weight": ("F32", (4, 8))})
    config = {"hidden_size": 8, "num_experts": 2}
    views, _ = plan_views(entries, config, all_experts=True)
    tensors = {name: torch.arange(torch.tensor(entry.shape).prod().item()).reshape(entry.shape).float() for name, entry in entries.items()}
    for view in views:
        full = decode_view(view, reader(tensors), config)
        block = decode_view(view, reader(tensors), config, row_slice=slice(1, 3))
        torch.testing.assert_close(block, full[1:3])


def test_embedded_lora_and_base_have_unique_identities():
    entries = E(**{
        "model.layers.0.q_proj.weight": ("BF16", (64, 64)),
        "base_model.model.model.layers.0.q_proj.lora_A.weight": ("BF16", (8, 64)),
        "base_model.model.model.layers.0.q_proj.lora_B.weight": ("BF16", (64, 8)),
    })
    views, coverage = plan_views(entries, {})
    assert len(views) == len({v.name for v in views}) == 2
    assert {v.name for v in views} == {
        "model.layers.0.q_proj.weight", "model.layers.0.q_proj.weight[lora]"}
    assert all(c["status"] == "measured" for c in coverage)
    standalone, _ = plan_views({k: v for k, v in entries.items() if "lora_" in k}, {})
    assert standalone[0].name == "model.layers.0.q_proj.weight"
