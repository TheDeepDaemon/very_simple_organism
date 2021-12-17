#include"cpp_headers.h"
#include"markov_clustering.h"
#include"grouping.h"
#include"data_augmentation.h"
#include<random>


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
        int64 rowSize = wCols * wRows;
        
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
        float* dataPtr, int64 rows, int64 cols, 
        int* outputGroups, unsigned int maxGroups, int iterations, double power, double inflation) {
        
        Eigen::Map<dmat> data(dataPtr, rows, cols);
        vector<set<int64>> groups = groupData(data, iterations, power, inflation);
        
        size_t numGroups = groups.size();
        set<int64>::iterator it;
        size_t j;
        size_t stopI = min((size_t)maxGroups, groups.size());
        for (size_t i = 0; i < stopI; i++) {
            for (it = groups[i].begin(), j = 0; it != groups[i].end(); ++it, j++) {
                outputGroups[(i * rows) + j] = (int)(*it);
            }
        }
        
    }
    
    
    // remove rows that do not fit in the cluster
    void removeOutliers(float* dataPtr, int64 rows, int64 cols, bool* keep) {
        Eigen::Map<dmat> data(dataPtr, rows, cols);
        
        vector<int64> group = getIndicesInGroup(data);
        
        for (const int64& memberInd: group) {
            keep[memberInd] = true;
        }
        
    }
    
    
    // remove things that are sufficiently similar.
    // indicate the result using an array of bools called "keep";
    // the bool value corresponding to a row tells whether to 
    // keep or remove. 
    void removeSimilar(float* dataPtr, int64 rows, int64 cols, bool* keep, float threshold) {
        
        Eigen::Map<dmat> data(dataPtr, rows, cols);
        memset(keep, true, rows);
        
        for (int64 i = 0; i < rows; i++) {
            for (int64 j = 0; j < rows; j++) {
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
    void scaleUp(float* dataPtr, int64 size) {
        Eigen::Map<evec> data(dataPtr, size);
        float maxVal = data.maxCoeff();
        data *= (1.0f / maxVal);
    }
    
    
    // get an average of a series of numpy arrays
    void averageArr(float* dataPtr, int64 n, int64 arrSize, float* outputAvePtr) {
        Eigen::Map<dmat> data(dataPtr, n, arrSize);
        Eigen::Map<evec> ave(outputAvePtr, arrSize);
        ave.setZero();
        
        for (int64 i = 0; i < data.rows(); i++) {
            ave += data.row(i);
        }
        
        ave /= (double)data.rows();
    }
    
    
    // normalized dot product: how much do these vectors 
    // point in the same direction?
    float normalizedDotProd(float* arrPtr1, float* arrPtr2, int64 size) {
        Eigen::Map<evec> arr1(arrPtr1, size);
        Eigen::Map<evec> arr2(arrPtr2, size);
        float ndp = arr1.normalized().dot(arr2.normalized());
        return ndp;
    }
    
    
    // calculate how close the values are to 0.5
    float calcEntropy(float* arrPtr, int64 size) {
        return calcEntropy(Eigen::Map<evec>(arrPtr, size));
    }
    
    
    // augment the data so that there is new data that is rotated and flipped
    void augmentData(float* dataPtr, 
	    const int64 n, const int64 rows, const int64 cols, 
	    float* augmentedDataPtr) {
        if (rows == cols) {
            augmentData<true>(dataPtr, n, rows, cols, augmentedDataPtr);
        }
        else {
            augmentData<false>(dataPtr, n, rows, cols, augmentedDataPtr);
        }
    }
    
    
    // find above or below average cutoff
    int64 findCutoff(float* data, int64 size) {
        float ave = 0;
        for (int64 i = 0; i < size; i++) {
            ave += data[i];
        }
        ave /= size;
        
        for (int64 i = 0; i < size; i++) {
            if (data[i] > ave) {
                return i;
            }
        }
        return size;
    }
    
    
    void centerData(float* dataPtr, int64 size) {
        float ave = 0;
        for (int64 i = 0; i < size; i++) {
            ave += dataPtr[i];
        }
        ave /= float(size);
        for (int64 i = 0; i < size; i++) {
            dataPtr[i] -= ave;
        }
    }
    
    
    void rowToWeights(float* dataPtr, int64 size) {
        
        float posSum = 0.0f;
        float negSum = 0.0f;

        for (int64 i = 0; i < size; i++) {
            float num = dataPtr[i];
            if (num > 0) {
                posSum += num;
            }
            else {
                negSum -= num;
            }
        }

        if (posSum == 0.0f) {
            posSum = 1.0f;
        }
        if (negSum == 0.0f) {
            negSum = 1.0f;
        }
        
        for (int64 i = 0; i < size; i++) {
            if (dataPtr[i] > 0) {
                dataPtr[i] /= posSum;
            }
            else {
                dataPtr[i] /= negSum;
            }
        }
        
    }
    
    
    // this function is here to convert the patterns collected into 
    // the values of the weights.
    // the dot product with a vector of ones should be zero
    void toWeights(float* dataPtr, int64 size, int64 rowSize) {
        for (int64 i = 0; i < size; i++) {
            float* data = dataPtr + (i * rowSize);
            centerData(data, rowSize);
            rowToWeights(data, rowSize);
        }
    }
    
}

