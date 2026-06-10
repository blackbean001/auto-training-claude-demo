import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import sys

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print('python3 quantize.py FROM_FILE TO_FILE',file=sys.stderr)
        sys.exit(1)
    quantize_dynamic(sys.argv[1], sys.argv[2], weight_type=QuantType.QUInt8, per_channel=True)

