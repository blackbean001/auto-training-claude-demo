import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("TRAJ_GPU",""))

import tensorflow as tf
import tf2onnx
import json
import argparse
import math

# 注: 原来这里有 `sys.path.append("../"); import dataset; import cnn_gru_v6`
# 这三行是没用到的死 import (本文件自带 build_infer_model_testing),
# 从项目根经 run_round 跑时会 ImportError → 已删除。


def load_model_testing(directory):
    with open(os.path.join(directory, 'options.json')) as reader:
        options = json.load(reader)
        model = build_model_testing(options)
        model.load_weights(os.path.join(directory, 'model'))
        @tf.function(input_signature=(tf.TensorSpec((None, None, options['dim_feature']), tf.dtypes.float32),))
        def infer(x):
            prediction, predicted_time = model([x])
            return prediction, predicted_time
        return model, infer, options


def build_model_testing(options):
    model = build_infer_model_testing(dim_feature=options['dim_feature'], dim_rnns=options['dim_rnns'])
    return model


def build_infer_model_testing(dim_feature, dim_rnns):
    return model


def export_model(model_file, exported_dir):
    # 注: 下面这个 options dict 是死代码 (dim_feature 实际从 ckpt 的 options.json 读),
    # 保留只为不动原结构; load_model_testing 会用真实 options。
    model_testing, _, _ = load_model_testing(model_file)
    model_testing.save(exported_dir)
    return model_testing


def convert_tflite(saved_model, exported_tflite):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(exported_tflite, "wb") as f:
        f.write(tflite_model)


def convert_onnx_buffer(model, onnx_dir):
    spec = (tf.TensorSpec((None, None, 2), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx_buffer = onnx_model.SerializeToString()
    with open(onnx_dir + "/model_buffer.onnx", "wb") as f:
        f.write(onnx_buffer)
    data = open(onnx_dir + "/model_buffer.onnx", "rb").read()
    with open(onnx_dir + "/onnx_model_buffer.h", "w") as f:
        f.write("const unsigned char model[] = {")
        for b in data:
            f.write(f"0x{b:02x},")
        f.write("};\n")
        f.write(f"const size_t model_len = {len(data)};\n")


if __name__ == "__main__":
    # ── 修复: 接收命令行 model_file (run_round 会传本轮 $SAVE_DIR) ──
    # 用法: python export_onnx_buffer.py <MODEL_FILE>
    # 不传参时退回旧的写死路径, 保证手动跑仍可用。
    ap = argparse.ArgumentParser(
        description="Export a trained ckpt dir to saved_model/onnx/quant.onnx/ort/header.")
    ap.add_argument(
        "model_file", nargs="?",
        default="v78_tmp/0605/finetune_full_tcl_reduce_fast_valid_fast/ckpt_ep4_u86000",
        help="待导出的 best-ckpt 目录 (含 options.json + model 权重)。")
    model_file = ap.parse_args().model_file

    exported_dir = model_file + "/saved_model"
    onnx_dir = model_file

    print(f"[export] model_file = {model_file}")
    model_testing = export_model(model_file, exported_dir)

    os.system(f"python -m tf2onnx.convert --saved-model {model_file}/saved_model --output {model_file}/model_hossom.onnx --opset 13")
    os.system(f"python quantize_onnx.py {model_file}/model_hossom.onnx {model_file}/model_hossom.quant.onnx")
    os.system(f"python -m onnxruntime.tools.convert_onnx_models_to_ort {model_file}/model_hossom.quant.onnx --optimization_style Runtime")

    data = open(f"{model_file}/model_hossom.quant.with_runtime_opt.ort", "rb").read()
    hex_array = ', '.join([f"0x{b:02x}" for b in data])

    with open(f"{model_file}/model_buffer_quant.with_runtime_opt.h", "w") as f:
        f.write("const unsigned char kerasbaseicortbuffer[] = {")
        f.write(hex_array)
        f.write("};\n")
        f.write(f"const size_t kerasbaseicortbuffer_size = {len(data)};")




