"""Prepare fixed-mode HC comparison sources, never production dispatch flags.

Apply to a private, stopped source snapshot only. Both variants suppress the
sidecar AND legacy RTN conversion. BF16 owns a separate immutable mapping;
the main checkpoint index and every non-HC tensor remain unchanged.
"""
from pathlib import Path


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError('source anchor changed: ' + str(path))
    path.write_text(text.replace(old, new))


def patch(snapshot, mode):
    if mode not in ('f16', 'bf16'):
        raise ValueError('fixed comparison mode required')
    root = Path(snapshot) / 'src/models/qwen38_flash_next'
    loader = root / 'exl3_weights.cpp'
    replace(loader, '    PrepareExl3HcQ8(storage, w.config, path);',
            '    // Comparison-only binary: retain original HC without Q8 sidecar.\n')
    replace(root / 'runtime/device_model.cpp',
            '  DeviceMixer Mixer(const HcMixer& m) {',
            '''  DeviceMixer Mixer(const HcMixer& m) {
    // Comparison-only EXL3 policy: also bypass automatic RTN Q8 copies.
    if (exl3_model)
      return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject), {}};''')
    replace(loader, '#include <cmath>', '#include <cmath>\n#include <cstdio>')
    hook = '''    if (m.down.type != core::GgmlType::kF16 ||
        m.up.type != core::GgmlType::kF16)
      Fail("comparison expects audited F16 HC source");
'''
    if mode == 'bf16':
        hook += '''    // Fixed comparison mount. Lifetime includes both trunk and MTP.
    static const auto original = ex::Checkpoint::Open("/hc-original");
    const auto bind_original = [&](TensorRef& ref) {
      const auto* t = original.Find(ref.name);
      if (!t || t->dtype != ex::DType::kBF16 ||
          t->shape != std::vector<std::int64_t>{ref.rows, ref.cols})
        Fail("invalid original BF16 HC " + std::string(ref.name));
      ref.data = t->data;
      ref.type = core::GgmlType::kBF16;
      ref.safetensors = false;
      ref.memory_owned = true;
    };
    bind_original(m.down);
    bind_original(m.up);
'''
        replace(root / 'runtime/executor_hc.cpp',
                'half_src_ == s_.xn && half_rows_ == n_tokens && half_cols_ == c.HcDim();',
                '''m.down.type == GgmlType::kF16 && !half_bf16_ &&
        half_src_ == s_.xn && half_rows_ == n_tokens && half_cols_ == c.HcDim();''')
    hook += '    std::fprintf(stderr, "HC precision ' + mode + ' pair: %s\\n", prefix.c_str());\n'
    replace(loader, '    if (storage.hc_q8) {', hook + '    if (storage.hc_q8) {')
