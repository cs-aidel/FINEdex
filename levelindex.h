#ifndef __LEVELINDEX_H__
#define __LEVELINDEX_H__

#include "util.h"

namespace aidel{

template<class key_t>
class LevelIndex {
    
public:
    explicit inline LevelIndex()
        : page_size(64), top_size(0), capacity(0), keys(nullptr), tables(nullptr) {}

    explicit inline LevelIndex(std::vector<key_t> &data)
        : page_size(64), keys(nullptr), tables(nullptr)
    {
        capacity = data.size();
        stride = page_size * page_size;
        assert(capacity>0);
        //copy data
        int hang = data.size() % page_size;
        hang = hang>0 ? page_size - hang : 0;
        keys = (key_t *) calloc((size_t)(capacity+hang), sizeof(key_t));
        for(size_t i=0; i<capacity; i++){
            keys[i] = data[i];
        }
        for(size_t i=capacity; i<(capacity+hang); i++){
            keys[i] = sizeof(key_t)==4? INT_MAX : LONG_MAX;
        }
        // construct upper levels
        if(capacity <= page_size){
            COUT_THIS("[Root-0level], capacity:" << capacity);
        }else if(capacity <= stride){
            onelevel(hang);
        }else{
            twolevel(hang);
        }
    }
    
    ~LevelIndex(){
        if(keys)
            free(keys);
        if(tables)
            free(tables);
        keys = nullptr;
        tables = nullptr;
    }

    size_t find(const key_t key){
        if(capacity <= page_size){
            return find_lower_avx(keys, page_size, key);
        }
        if(capacity<=stride)
            return find_1level(key);
        return find_2level(key);
    }


private:
    void onelevel(size_t hang){
        top_size = (int) std::ceil((capacity+hang) / double(page_size));
        COUT_THIS("[Root-1level] capacity+hang:"<<capacity+hang <<" ,top_size: " << top_size);
        tables = (key_t*) calloc((size_t) top_size, sizeof(key_t));
        // populate top level index
        for (int i = 0; i < top_size; i++) {
            size_t pos = (i + 1) * page_size;
            if(pos<=capacity){
                tables[i] = keys[pos-1];
            }else{
                tables[i] = sizeof(key_t)==4? INT_MAX : LONG_MAX;
            }
        }
    }
    void twolevel(size_t hang){
        top_size = (int) std::ceil((capacity+hang) / double(page_size * page_size));
        COUT_THIS("[Root-2level] top_size: " << top_size);
        tables = (key_t*) calloc((size_t) top_size * (page_size + 1), sizeof(key_t));
        // populate second level index
        for (int i = 0; i < top_size * page_size; i++) {
            size_t pos = (i + 1) * page_size;
            if(pos<=capacity){
                tables[top_size + i] = keys[pos-1];
            }else{
                tables[top_size + i] = sizeof(key_t)==4? INT_MAX : LONG_MAX;
            }
        }
        // populate top level index
        for (int i = 0; i < top_size; i++) {
            tables[i] = tables[top_size + (i + 1) * page_size - 1];
        }
    }
    size_t find_1level(const key_t key) {
        int i = find_lower(tables, top_size, key);
        int pos = i*page_size;
        memory_fence();
        //int offset = find_lower_avx(keys + pos, page_size, key);
        int offset = find_lower(keys + pos, page_size, key);
        return pos + offset;
    }
    size_t find_2level(const key_t key) {
        int i = find_lower(tables, top_size, key);
        memory_fence();
        //int j = find_lower_avx(tables + top_size + i * page_size, page_size, key);
        int j = find_lower(tables + top_size + i * page_size, page_size, key);
        int pos = i * stride + j * page_size;
        memory_fence();
        //int offset = find_lower_avx(keys + pos, page_size, key);
        int offset = find_lower(keys + pos, page_size, key);
        return pos + offset;
    }
    inline size_t find_lower_avx(const int *arr, int n, int key){
        __m256i vkey = _mm256_set1_epi32(key);
        __m256i cnt = _mm256_setzero_si256();
        for (int i = 0; i < n; i += 8) {
            __m256i mask0 = _mm256_cmpgt_epi32(vkey, _mm256_load_si256((__m256i *)&arr[i+0]));
            cnt = _mm256_sub_epi32(cnt, mask0);
        }
        __m128i xcnt = _mm_add_epi32(_mm256_extracti128_si256(cnt, 1), _mm256_castsi256_si128(cnt));
        xcnt = _mm_add_epi32(xcnt, _mm_shuffle_epi32(xcnt, SHUF(2, 3, 0, 1)));
        xcnt = _mm_add_epi32(xcnt, _mm_shuffle_epi32(xcnt, SHUF(1, 0, 3, 2)));

        return _mm_cvtsi128_si32(xcnt);
    }
    inline size_t find_lower_avx(const int64_t *arr, int n, int64_t key) {
        __m256i vkey = _mm256_set1_epi64x(key);
        __m256i cnt = _mm256_setzero_si256();
        for (int i = 0; i < n; i += 8) {
          __m256i mask0 = _mm256_cmpgt_epi64(vkey, _mm256_load_si256((__m256i *)&arr[i+0]));
          __m256i mask1 = _mm256_cmpgt_epi64(vkey, _mm256_load_si256((__m256i *)&arr[i+4]));
          __m256i sum = _mm256_add_epi64(mask0, mask1);
          cnt = _mm256_sub_epi64(cnt, sum);
        }
        __m128i xcnt = _mm_add_epi64(_mm256_extracti128_si256(cnt, 1), _mm256_castsi256_si128(cnt));
        xcnt = _mm_add_epi64(xcnt, _mm_shuffle_epi32(xcnt, SHUF(2, 3, 0, 1)));
        return _mm_cvtsi128_si32(xcnt);
    }
    inline size_t find_lower(const key_t *arr, int n, key_t key) {
        intptr_t pos = -1;
        intptr_t logstep = bsr(n - 1);
        intptr_t step = intptr_t(1) << logstep;

        pos = (arr[pos + n - step] < key ? pos + n - step : pos);
        step >>= 1;

        while (step > 0) {
          pos = (arr[pos + step] < key ? pos + step : pos);
          step >>= 1;
        }
        pos += 1;

        return (int) (arr[pos] >= key ? pos : n);
    }


private:
    size_t page_size = 64;
    size_t stride = 0;
    size_t top_size = 0;
    size_t capacity = 0;
    size_t level = 0;
    key_t *keys = nullptr;
    key_t *tables = nullptr;

};

class TwoLevelIndex {
    const int* data_;
    int k1_, k2_, k2stride_, page_size_;
    int* tables_;

public:
    explicit TwoLevelIndex(std::vector<int>& data, int k2, int page_size)
            : k2_(k2), page_size_(page_size) {
        int hang = data.size() % page_size;
        if (hang > 0) {
            for (int i = 0; i < page_size - hang; i++) {
                data.push_back(INT_MAX);
            }
        }

        data_ = &data[0];

        k1_ = (int) std::ceil(data.size() / double(k2_ * page_size_));
        std::cerr << "2-level index top level size = " << k1_ << std::endl;
        tables_ = (int*) calloc((size_t) k1_ * (k2 + 1), sizeof(int));
        k2stride_ = k2 * page_size_;

        // populate second level index
        for (int i = 0; i < k1_ * k2_; i++) {
            tables_[k1_ + i] = ((i + 1) * page_size_ > data.size()) ? INT_MAX : data[(i + 1) * page_size_ - 1];
        }
        // populate top level index
        for (int i = 0; i < k1_; i++) {
            tables_[i] = tables_[k1_ + (i + 1) * k2 - 1];
        }  
    }

    ~TwoLevelIndex() {
        free(tables_);
    }

    // assumes that key is <= max(data)
    int find(int key) {
        int i = binary_search_branchless(tables_, k1_, key);
        //int i = interpolation_search(tables_, k1_, key);
        //int i = linear_search_avx(tables_, k1_, key);
        //int j = binary_search_branchless(tables_ + k1_ + i * k2_, k2_, key);
        int j = linear_search_avx(tables_ + k1_ + i * k2_, k2_, key);
        int pos = i * k2stride_ + j * page_size_;
        int offset = linear_search_avx(data_ + pos, page_size_, key);
        return pos + offset;
    }
};


class ThreeLevelIndex {
  const int* data_;
  int k1_, k2_, k3_, stride1_, stride2_, page_size_;
  int* tables_;
  int* table2_;
  int* table3_;

 public:
    explicit ThreeLevelIndex(std::vector<int>& data, int k2, int k3, int page_size)
            : k2_(k2), k3_(k3), page_size_(page_size) {

        int hang = data.size() % page_size;
        if (hang > 0) {
            for (int i = 0; i < page_size - hang; i++) {
                data.push_back(INT_MAX);
            }
        }

        data_ = &data[0];

        k1_ = (int) std::ceil(data.size() / double(k2_ * k3_ * page_size_));
        std::cerr << "3-level index top level size = " << k1_ << std::endl;
        tables_ = (int*) calloc((size_t) k1_ * (1 + k2_ * (1 + k3_)), sizeof(int));
        table2_ = tables_ + k1_;
        table3_ = table2_ + k1_ * k2_;

        stride1_ = k2_ * k3_ * page_size_;
        stride2_ = k3_ * page_size_;

        // populate third level index
        for (int i = 0; i < k1_ * k2_ * k3_; i++) {
          table3_[i] = ((i + 1) * page_size_ > data.size()) ? INT_MAX : data[(i + 1) * page_size_ - 1];
        }

        // populate second level index
        for (int i = 0; i < k1_ * k2_; i++) {
            table2_[i] = table3_[(i + 1) * k3_ - 1];
        }

        // populate top level index
        for (int i = 0; i < k1_; i++) {
            tables_[i] = table2_[(i + 1) * k2_ - 1];
        }
    }

    ~ThreeLevelIndex() {
        free(tables_);
    }

    // assumes that key is <= max(data)
    int find(int key) {
        int i = binary_search_branchless(tables_, k1_, key);
        //int i = interpolation_search(tables_, k1_, key);
        //int i = linear_search_avx(tables_, k1_, key);
        int j = linear_search_avx(table2_ + i * k2_, k2_, key);
        int k = linear_search_avx(table3_ + i * k2_ * k3_ + j * k3_, k3_, key);
        int pos = i * stride1_ + j * stride2_ + k * page_size_;
        int offset = linear_search_avx(data_ + pos, page_size_, key);
        return pos + offset;
    }
};



} // namespace aidel



#endif