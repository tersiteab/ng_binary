#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <dlfcn.h> // If you use dynamic loading

// Forward declare your external functions (if statically linked, just #include the header)
extern "C" {
    void createPillars(
        const float* points, int num_points,
        float* tensor_out, int* indices_out,
        int maxPointsPerPillar,
        int maxPillars,
        float xStep,
        float yStep,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        float zMin,
        float zMax,
        bool printTime
    );

    void createPillarsTargetC(
        const float* objectPositions,
        const float* objectDimensions,
        const float* objectYaws,
        const int* objectClassIds,
        const float* anchorDimensions,
        const float* anchorZHeights,
        const float* anchorYaws,
        float positiveThreshold,
        float negativeThreshold,
        int nbObjects,
        int nbAnchors,
        int nbClasses,
        int downscalingFactor,
        float xStep,
        float yStep,
        float xMin,
        float xMax,
        float yMin,
        float yMax,
        float zMin,
        float zMax,
        int xSize,
        int ySize,
        float* tensor_out,
        int* posCnt_out,
        int* negCnt_out,
        bool printTime
    );
}

int main() {
    constexpr int N = 100; // Number of points
    constexpr int maxPointsPerPillar = 10;
    constexpr int maxPillars = 20;
    constexpr float xMin = 0.0f, xMax = 50.0f;
    constexpr float yMin = 0.0f, yMax = 50.0f;
    constexpr float zMin = -3.0f, zMax = 1.0f;
    constexpr float xStep = 0.5f;
    constexpr float yStep = 0.5f;

    // Step 1: Generate N x 4 points (x, y, z, intensity)
    std::vector<float> points(N * 4);
    for (int i = 0; i < N; ++i) {
        points[i * 4 + 0] = static_cast<float>(rand() % 50);      // x
        points[i * 4 + 1] = static_cast<float>(rand() % 50);      // y
        points[i * 4 + 2] = static_cast<float>((rand() % 5) - 3); // z (-3 to +1)
        points[i * 4 + 3] = static_cast<float>(rand()) / RAND_MAX; // intensity
    }

    // Step 2: Allocate output arrays
    constexpr int tensorSize = 1 * maxPillars * maxPointsPerPillar * 7;
    constexpr int indicesSize = 1 * maxPillars * 3;

    std::vector<float> tensor_out(tensorSize, 0);
    std::vector<int> indices_out(indicesSize, 0);

    // Step 3: Call the function
    createPillars(
        points.data(), N,
        tensor_out.data(), indices_out.data(),
        maxPointsPerPillar, maxPillars,
        xStep, yStep,
        xMin, xMax,
        yMin, yMax,
        zMin, zMax,
        true // printTime
    );

    // Step 4: Print some output for verification
    std::cout << "Sample tensor output (first 10 elements):" << std::endl;
    for (int i = 0; i < std::min(tensorSize, 10); ++i) {
        std::cout << tensor_out[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Sample indices output:" << std::endl;
    for (int i = 0; i < std::min(indicesSize, 9); i += 3) {
        std::cout << "(" << indices_out[i] << ", " << indices_out[i + 1] << ", " << indices_out[i + 2] << ") ";
    }
    std::cout << "\n";

    return 0;
}
