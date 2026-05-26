import io
import struct
import torch
from PIL import Image

class CasiaWebface(torch.utils.data.Dataset):
    def __init__(self, rec_path, idx_path):
        super().__init__()
        # All records
        self.rec_path = rec_path
        # Index
        self.idx_path = idx_path
        self.idx_map = self.read_index()

    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()

    def __getitem__(self, item):
        # Open file if needed
        if not hasattr(self, 'f'):
            self.f = open(self.rec_path, 'rb')
        # Get file offset
        offset = self.idx_map[item]
        self.f.seek(offset)
        # 8 info bytes (4 -> kmagic, 4 -> lrecord)
        info_bytes = self.f.read(8)
        kmagic, lrecord = struct.unpack('2I', info_bytes)
        # lrecord contains 29 last bits for length of image
        record_length = lrecord & ((1 << 29) - 1)
        # 24 bytes for flag, label and etc.
        header = self.f.read(24)
        flag, label = struct.unpack('If', header[:8])
        # Read image
        image_bytes = self.f.read(record_length)
        image_object = io.BytesIO(image_bytes)
        image = Image.open(image_object).convert('RGB')
        return image, int(label)

    def __len__(self):
        # Images after 490 622 are broken
        # return len(self.idx_map.keys())
        return 490623

    def read_index(self):
        index_map = {}
        with open(self.idx_path, 'r') as f:
            # For each row get (key, offset)
            for line in f.readlines():
                if not line.strip():
                    continue
                key, offset = list(map(int, line.split()))
                # Keys start from 1
                index_map[int(key) - 1] = offset
        return index_map