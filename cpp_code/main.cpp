#include"cpp_headers.h"
#include"markov_clustering.h"
#include"grouping.h"
#include"data_augmentation.h"



extern "C" {
    
    // apply this (modified) sigmoid function to whole vector.
    // designed for 0 to 1 values. moves values that are
    // less than 0.5 closer to 0, and the values greater
    // than 0.5 closer to 1
    void applySigmoid(float* arr, size_t size) {
        for (size_t i = 0; i < size; i++) {
            arr[i] = sigmoid(arr[i], 0.5f, 16);
        }
    }
    
    // takes images and extracts patches from them, 
    // stores the patches as rows in a matrix
    void imagesToMatrix(float* destination, float* srcImgPtr, size_t numEntries, 
        size_t cols, size_t rows, size_t wCols, size_t wRows) {
        
        size_t jn = cols - wCols + 1;
        size_t kn = rows - wRows + 1;
        int64_t rowSize = wCols * wRows;
        
        size_t numI = jn * kn;

        for (size_t i = 0; i < numEntries; i++) {
            size_t iImageOffset = (i * cols * rows);
            size_t iMatOffset = (i * numI);

            for (size_t j = 0; j < jn; j++) {
                // the offset of the pointer within the matrix
                size_t jImageOffset = (j * rows);
                
                // the ptr offset of the current matrix
                size_t jMatOffset = (j * kn);

                for (size_t k = 0; k < kn; k++) {
                    for (size_t l = 0; l < wCols; l++) {
                        // get the location for the image
                        size_t imgOffset = iImageOffset + jImageOffset + k + (l * rows);
                        float* imgPtr = srcImgPtr + imgOffset;
                        
                        // get the location for the matrix
                        size_t rowNumber = iMatOffset + jMatOffset + k;
                        size_t matOffset = (rowNumber * rowSize) + (l * wRows);
                        
                        // copy the column from the image to the matrix
                        memcpy(destination + matOffset, imgPtr, wRows * sizeof(float));
                    }
                }
            }
        }
    }
    
    // find groups, output a indices that represent members of each group
    void findGroups(
        float* dataPtr, int64_t rows, int64_t cols, 
        int* outputGroups, unsigned int maxGroups, int iterations, double power, double inflation) {
        
        Eigen::Map<dmat> data(dataPtr, rows, cols);
        vector<set<int64_t>> groups = groupData(data, iterations, power, inflation);
        
        size_t numGroups = groups.size();
        set<int64_t>::iterator it;
        size_t j;
        size_t stopI = min((size_t)maxGroups, groups.size());
        for (size_t i = 0; i < stopI; i++) {
            for (it = groups[i].begin(), j = 0; it != groups[i].end(); ++it, j++) {
                outputGroups[(i * rows) + j] = (int)(*it);
            }
        }
        
    }
    
    // remove rows that do not fit in the cluster
    void removeOutliers(float* dataPtr, int64_t rows, int64_t cols, bool* keep) {
        Eigen::Map<dmat> data(dataPtr, rows, cols);
        
        vector<int64_t> group = getIndicesInGroup(data);
        
        for (const int64_t& memberInd: group) {
            keep[memberInd] = true;
        }
        
    }
    
    // revise this function
    void joinSimilar(float* dataPtr, int64_t rows, int64_t cols, bool* keep, float threshold) {
        
        Eigen::Map<dmat> data(dataPtr, rows, cols);
        memset(keep, true, rows);
        
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < rows; j++) {
                if (i != j && keep[i]) {
                    float diff = (data.row(i) - data.row(j)).norm();
                    if (diff < threshold) {
                        keep[j] = false;
                    }
                }
            }
        }
        
    }
    
    // scale all values so that the largest value is 1.0
    void scaleUp(float* dataPtr, int64_t size) {
        Eigen::Map<evec> data(dataPtr, size);
        float maxVal = data.maxCoeff();
        data *= (1.0f / maxVal);
    }
    
    // get an average of a series of numpy arrays
    void averageArr(float* dataPtr, int64_t n, int64_t arrSize, float* outputAvePtr) {
        Eigen::Map<dmat> data(dataPtr, n, arrSize);
        Eigen::Map<evec> ave(outputAvePtr, arrSize);
        ave.setZero();
        
        for (int64_t i = 0; i < data.rows(); i++) {
            ave += data.row(i);
        }
        
        ave /= (double)data.rows();
    }
    
    // normalized dot product: how much do these vectors 
    // point in the same direction?
    float normalizedDotProd(float* arrPtr1, float* arrPtr2, int64_t size) {
        Eigen::Map<evec> arr1(arrPtr1, size);
        Eigen::Map<evec> arr2(arrPtr2, size);
        float ndp = arr1.normalized().dot(arr2.normalized());
        return ndp;
    }
    
    float calcEntropy(float* arrPtr, int64_t size) {
        return calcEntropy(Eigen::Map<evec>(arrPtr, size));
    }
    
    void augmentData(float* dataPtr, 
	    const int64_t n, const int64_t rows, const int64_t cols, 
	    float* augmentedDataPtr) {
        if (rows == cols) {
            augmentData<true>(dataPtr, n, rows, cols, augmentedDataPtr);
        }
        else {
            augmentData<false>(dataPtr, n, rows, cols, augmentedDataPtr);
        }
    }
    
}

