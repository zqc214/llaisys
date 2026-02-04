#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // 空张量或标量视为连续
    if (_meta.shape.empty()) {
        return true;
    }
    
    // 从最后一维开始检查，期望的步长从 1 开始
    ptrdiff_t expected_stride = 1;
    for (size_t i = _meta.shape.size(); i > 0; --i) {
        size_t dim = i - 1;
        // 如果该维度大小为1，步长可以是任意值（不影响连续性）
        if (_meta.shape[dim] != 1) {
            if (_meta.strides[dim] != expected_stride) {
                return false;
            }
        }
        expected_stride *= _meta.shape[dim];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 1. 检查 order 的维度数是否正确
    CHECK_ARGUMENT(order.size() == this->ndim(), "permute order size must match tensor dimensions");
    
    // 2. 检查 order 是否是有效的排列（0 到 ndim-1 各出现一次）
    std::vector<bool> seen(this->ndim(), false);
    for (size_t idx : order) {
        CHECK_ARGUMENT(idx < this->ndim(), "permute index out of range");
        CHECK_ARGUMENT(!seen[idx], "permute order contains duplicate indices");
        seen[idx] = true;
    }
    
    // 3. 按照 order 重新排列 shape 和 strides
    std::vector<size_t> new_shape(this->ndim());
    std::vector<ptrdiff_t> new_strides(this->ndim());
    for (size_t i = 0; i < this->ndim(); ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    
    // 4. 创建新的元数据
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    
    // 5. 返回共享相同 storage 的新张量
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1. 计算新形状的元素总数
    size_t new_numel = 1;
    for (size_t s : shape) {
        new_numel *= s;
    }
    
    // 2. 检查元素总数是否一致
    CHECK_ARGUMENT(new_numel == this->numel(), "view size mismatch: total elements must be the same");
    
    // 3. 只有连续张量才能进行 view
    // 非连续张量需要先调用 contiguous() 再 view
    CHECK_ARGUMENT(this->isContiguous(), "view requires a contiguous tensor");
    
    // 4. 计算新的 strides（按连续布局）
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        new_strides[i - 1] = stride;
        stride *= shape[i - 1];
    }
    
    // 5. 创建新的元数据
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    
    // 6. 返回共享相同 storage 的新张量
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 1. 检查 dim 是否在范围内
    CHECK_ARGUMENT(dim < this->ndim(), "slice dimension out of range");
    
    // 2. 检查 start 和 end 是否有效
    CHECK_ARGUMENT(start <= end, "slice start must be <= end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice end out of range");
    
    // 3. 计算新的 shape
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;
    
    // 4. strides 保持不变
    std::vector<ptrdiff_t> new_strides = _meta.strides;
    
    // 5. 计算新的 offset（字节偏移）
    // 新 offset = 原 offset + start * strides[dim] * elementSize
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize();
    
    // 6. 创建新的元数据
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    
    // 7. 返回共享相同 storage 的新张量
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    core::context().setDevice(this->deviceType(), this->deviceId());
    size_t bytes = this->numel() * this->elementSize();
    llaisysMemcpyKind_t kind = (this->deviceType() == LLAISYS_DEVICE_CPU) 
                               ? LLAISYS_MEMCPY_H2H 
                               : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(
        this->data(),    
        src_,            
        bytes,            
        kind               
    );
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
