import torch
from typing import Iterable, Sequence, Union

class TensorCache:
    def __init__(self, max_cache_length: int, dims: Sequence[int], device: torch.device, dtype=torch.float):
        self.shape = (max_cache_length, *dims)
        self.cache = torch.empty(self.shape, dtype=dtype, device=device)

    def put(self, index: int, tensor: torch.Tensor):
        self.cache[index, ...] = tensor

    def put_batch(self, indices: Sequence[int], tensor_list: Union[torch.Tensor, Iterable[torch.Tensor]]):
        assert len(indices) == len(tensor_list)
        if isinstance(tensor_list, torch.Tensor):
            self.cache[indices, ...] = tensor_list
        else:
            for index, tensor in zip(indices, tensor_list):
                self.cache[index, ...] = tensor

    def get_all(self):
        return self.cache

    def get_batch(self, indices: Sequence[int]):
        return self.cache[indices, ...]

    def get(self, index: int):
        return self.cache[index, ...]


class MultiScaleTensorCache:
    def __init__(self, max_num_elements: int, dims_list: Sequence[Sequence[int]], device: torch.device):
        self.shape_list = tuple((max_num_elements, *dims) for dims in dims_list)
        self.cache_list = tuple(torch.empty(shape, dtype=torch.float, device=device) for shape in self.shape_list)

    def put(self, index: int, multi_scale_tensor: Sequence[torch.Tensor]):
        assert len(multi_scale_tensor) == len(self.cache_list)
        for cache, tensor in zip(self.cache_list, multi_scale_tensor):
            cache[index, ...] = tensor

    def put_batch(self, indices: Sequence[int], multi_scale_tensor_list: Sequence[Union[torch.Tensor, Sequence[torch.Tensor]]]):
        assert len(multi_scale_tensor_list) == len(self.cache_list)
        for cache, tensor_list in zip(self.cache_list, multi_scale_tensor_list):
            assert len(indices) == len(tensor_list)
            if isinstance(tensor_list, torch.Tensor):
                cache[indices, ...] = tensor_list
            else:
                for index, tensor in zip(indices, tensor_list):
                    cache[index, ...] = tensor

    def get_all(self):
        return self.cache_list

    def get_batch(self, indices):
        return tuple(cache[indices, ...] for cache in self.cache_list)

    def get(self, index):
        return tuple(cache[index, ...] for cache in self.cache_list)

class CacheService:
    def __init__(self, max_num_elements, cache):
        self.id_list = [None] * max_num_elements
        self.free_bits = [True] * max_num_elements
        self.cache = cache

    def reset(self):
        # Reset the cache: mark all slots as free and clear the id list.
        self.id_list = [None] * len(self.id_list)
        self.free_bits = [True] * len(self.free_bits)
        # Optionally, clear the underlying cache if needed.
        # 예: self.cache.cache.zero_()

    def expand_cache(self):
        old_capacity = len(self.id_list)
        new_capacity = old_capacity * 2
        # 확장된 id_list와 free_bits 생성
        self.id_list.extend([None] * (new_capacity - old_capacity))
        self.free_bits.extend([True] * (new_capacity - old_capacity))
        # underlying TensorCache 확장
        # 기존 cache의 shape은 (old_capacity, *dims)입니다.
        dims = self.cache.cache.shape[1:]
        device = self.cache.cache.device
        dtype = self.cache.cache.dtype
        # 새 TensorCache 객체 생성
        new_cache = TensorCache(new_capacity, dims, device, dtype=dtype)
        # 기존 데이터를 복사합니다.
        new_cache.cache[:old_capacity, ...] = self.cache.cache
        self.cache = new_cache
        # print(f"[CacheService] Expanded cache from capacity {old_capacity} to {new_capacity}")


    def put(self, id_, item):
        try:
            index = self.id_list.index(id_)
        except ValueError:
            index = None
        if index is not None:
            # 이미 id가 존재하면 업데이트
            assert not self.free_bits[index]
        else:
            if True not in self.free_bits:
                # print("[CacheService] Warning: Cache is full. Expanding cache capacity instead of resetting.")
                self.expand_cache()
            # free slot을 찾습니다.
            index = self.free_bits.index(True)
            self.id_list[index] = id_
            self.free_bits[index] = False
        self.cache.put(index, item)
        
    def put_batch(self, ids, items):
        indices = []
        for id_ in ids:
            if id_ in self.id_list:
                index = self.id_list.index(id_)
                assert not self.free_bits[index]
            else:
                if True not in self.free_bits:
                    print("[CacheService] Warning: Cache is full. Resetting cache before batch put.")
                    self.reset()
                index = self.free_bits.index(True)
                self.id_list[index] = id_
                self.free_bits[index] = False
            indices.append(index)
        self.cache.put_batch(indices, items)

    def get_all(self):
        return self.cache.get_all()

    def delete(self, id_):
        index = self.id_list.index(id_)
        self.free_bits[index] = True
        self.id_list[index] = None

    def rename(self, old_id, new_id):
        assert new_id not in self.id_list
        index = self.id_list.index(old_id)
        self.id_list[index] = new_id

    def get(self, id_):
        try:
            index = self.id_list.index(id_)
        except ValueError:
            print(f"[CacheService] Warning: id {id_} not found in cache. Returning None.")
            return None
        return self.cache.get(index)

    def get_batch(self, ids):
        indices = [self.id_list.index(id_) for id_ in ids]
        return self.cache.get_batch(indices)
