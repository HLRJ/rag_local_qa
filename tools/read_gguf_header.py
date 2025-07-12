def read_gguf_arch(filename):
    with open(filename, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError("Not a GGUF file")
        f.read(4)  # version
        n_kv = int.from_bytes(f.read(8), 'little')
        for i in range(n_kv):
            key_len = int.from_bytes(f.read(8), 'little')
            if key_len > 256:
                print(f"[警告] 第 {i} 项 key_len 太大({key_len})，跳过")
                f.seek(key_len, 1)
                continue
            key = f.read(key_len).decode(errors='ignore')
            value_type = f.read(1)
            if value_type == b'\x01':  # string
                str_len = int.from_bytes(f.read(8), 'little')
                if str_len > 1024:
                    print(f"[警告] 第 {i} 项 string 太大({str_len})，跳过")
                    f.seek(str_len, 1)
                    continue
                value = f.read(str_len).decode(errors='ignore')
                print(f"[调试] 读取到 key={key}, value={value}")
                if key == 'general.architecture':
                    return value
            else:
                # 跳过其他类型
                if value_type == b'\x00':  # bool
                    f.read(1)
                elif value_type == b'\x02':  # uint8 array
                    count = int.from_bytes(f.read(8), 'little')
                    f.read(count)
                elif value_type == b'\x03':  # uint32 array
                    count = int.from_bytes(f.read(8), 'little')
                    f.read(count * 4)
                elif value_type == b'\x04':  # float32 array
                    count = int.from_bytes(f.read(8), 'little')
                    f.read(count * 4)
                else:
                    print(f"[调试] 未知类型: {value_type}")
                    return "unknown"
    return "unknown"

print("模型架构:", read_gguf_arch("F:/project/rag_local_qa/models/minicpm/MiniCPM-2B-128k-Q2_K.gguf"))


with open("F:/project/rag_local_qa/models/minicpm/MiniCPM-2B-128k-Q2_K.gguf", "rb") as f:
    header = f.read(16)
print("文件头:", header)



