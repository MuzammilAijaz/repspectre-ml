# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false

import logging
import os
from typing import Final

logger: Final = logging.getLogger(__name__)


def _convert_bytes_to_c_array(
    data: bytes,
    array_name: str = "g_model_data",
    include_header: str = "model_data.h",
) -> tuple[str, str]:
    """Converts raw bytes (such as a TFLite model) into C source and header definitions.

    Returns:
        tuple[header_content, source_content]
    """
    total_len = len(data)

    header = f"""#ifndef {array_name.upper()}_H_
#define {array_name.upper()}_H_

#ifdef __cplusplus
extern "C" {{
#endif

// 16-byte alignment required by TensorFlow Lite for Microcontrollers
extern const unsigned char {array_name}[];
extern const unsigned int {array_name}_len;

#ifdef __cplusplus
}}
#endif

#endif  // {array_name.upper()}_H_
"""

    # Format bytes in hex: 12 bytes per line
    hex_lines: list[str] = []
    chunk_size = 12
    for i in range(0, total_len, chunk_size):
        chunk = data[i : i + chunk_size]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"  {hex_str},")

    bytes_body = "\n".join(hex_lines)

    source = f"""#include "{include_header}"

// Align to 16 bytes for optimal ESP32 / TFLM SIMD memory alignment
__attribute__((aligned(16))) const unsigned char {array_name}[] = {{
{bytes_body}
}};

const unsigned int {array_name}_len = {total_len};
"""

    return header, source


def export_tflite_to_c_array(
    tflite_path: str,
    output_dir: str,
    array_name: str = "g_model_data",
    header_filename: str = "model_data.h",
    source_filename: str = "model_data.c",
) -> tuple[str, str]:
    """Reads a .tflite model file and exports C header and source files for ESP-IDF / TFLM.

    Parameters:
        tflite_path: Path to the .tflite binary file.
        output_dir: Directory where .h and .c files will be written.
        array_name: C identifier for the byte array.
        header_filename: Name of the header file to generate.
        source_filename: Name of the C source file to generate.

    Returns:
        tuple of (header_path, source_path)
    """
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model file not found: {tflite_path}")

    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    os.makedirs(output_dir, exist_ok=True)

    header_content, source_content = _convert_bytes_to_c_array(
        model_bytes,
        array_name=array_name,
        include_header=header_filename,
    )

    header_path = os.path.join(output_dir, header_filename)
    source_path = os.path.join(output_dir, source_filename)

    with open(header_path, "w") as f:
        f.write(header_content)

    with open(source_path, "w") as f:
        f.write(source_content)

    logger.info(
        "Exported ESP-IDF C model array (%d bytes): %s and %s",
        len(model_bytes),
        header_path,
        source_path,
    )

    return header_path, source_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export TFLite model to ESP-IDF C array")
    parser.add_argument("--tflite", type=str, default="models/export/model_quantized.tflite")
    parser.add_argument("--output-dir", type=str, default="models/export")
    args = parser.parse_args()

    export_tflite_to_c_array(args.tflite, args.output_dir)
