#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h>

using namespace std;

const float PI = 3.14159265359f;

struct Image
{
    int size;
    vector<float> data;
};

struct Sinogram
{
    int n_angles;
    int n_sensors;
    vector<float> data;
};

// --- Forward Projection (Radon Transform) ---
void forward_project(const Image &img, Sinogram &sino, const vector<float> &angles)
{
    int center = img.size / 2;

#pragma omp parallel for collapse(2)
    for (int a = 0; a < sino.n_angles; a++)
    {
        for (int s = 0; s < sino.n_sensors; s++)
        {
            float theta = angles[a];
            float cos_t = cos(theta);
            float sin_t = sin(theta);
            float sensor_pos = s - (sino.n_sensors / 2.0f);

            float sum = 0.0f;
            // Ray-driven line integration
            for (float t = -img.size; t < img.size; t += 0.5f)
            {
                float x = sensor_pos * cos_t - t * sin_t + center;
                float y = sensor_pos * sin_t + t * cos_t + center;

                int ix = (int)x;
                int iy = (int)y;

                if (ix >= 0 && ix < img.size && iy >= 0 && iy < img.size)
                {
                    sum += img.data[iy * img.size + ix];
                }
            }
            sino.data[a * sino.n_sensors + s] = sum * 0.5f; // Step size scaling
        }
    }
}

// --- Back Projection (Adjoint Operator) ---
void back_project(const Sinogram &sino, Image &img, const vector<float> &angles)
{
    int center = img.size / 2;
    fill(img.data.begin(), img.data.end(), 0.0f);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < img.size; y++)
    {
        for (int x = 0; x < img.size; x++)
        {
            float val = 0.0f;
            for (int a = 0; a < sino.n_angles; a++)
            {
                float dist = (x - center) * cos(angles[a]) + (y - center) * sin(angles[a]);
                int s = (int)(dist + sino.n_sensors / 2.0f);
                if (s >= 0 && s < sino.n_sensors)
                {
                    val += sino.data[a * sino.n_sensors + s];
                }
            }
            img.data[y * img.size + x] = val / sino.n_angles;
        }
    }
}

void save_binary(const Image &img, string path)
{
    ofstream out(path, ios::binary);
    out.write((char *)img.data.data(), img.data.size() * sizeof(float));
    out.close();
}

int main(int argc, char *argv[])
{
    // Usage: ./sirt_solver <input_phantom> <output_bin> <num_iterations>
    if (argc < 4)
    {
        cout << "Usage: " << argv[0] << " <input_bin> <output_bin> <num_iters>" << endl;
        return 1;
    }

    string input_path = argv[1];
    string output_path = argv[2];
    int max_iters = atoi(argv[3]);

    int size = 256;
    int n_angles = 180;
    int n_sensors = 362;

    // 1. Load Phantom
    Image phantom;
    phantom.size = size;
    phantom.data.resize(size * size);
    ifstream in(input_path, ios::binary);
    if (!in)
        return 1;
    in.read((char *)phantom.data.data(), size * size * sizeof(float));
    in.close();

    // 2. Setup Angles & Sinogram
    vector<float> angles(n_angles);
    for (int i = 0; i < n_angles; i++)
        angles[i] = i * 3.14159265359f / n_angles;

    Sinogram measured_sino;
    measured_sino.n_angles = n_angles;
    measured_sino.n_sensors = n_sensors;
    measured_sino.data.resize(n_angles * n_sensors);
    forward_project(phantom, measured_sino, angles);

    // 3. SIRT Algorithm
    Image recon;
    recon.size = size;
    recon.data.assign(size * size, 0.0f);
    Sinogram sim_sino = measured_sino;
    Image residual_img = recon;

    for (int iter = 1; iter <= max_iters; iter++)
    {
        forward_project(recon, sim_sino, angles);
        for (size_t i = 0; i < sim_sino.data.size(); i++)
        {
            sim_sino.data[i] = measured_sino.data[i] - sim_sino.data[i];
        }
        back_project(sim_sino, residual_img, angles);

#pragma omp parallel for
        for (size_t i = 0; i < recon.data.size(); i++)
        {
            recon.data[i] += 0.001f * residual_img.data[i];
            if (recon.data[i] < 0)
                recon.data[i] = 0;
        }
    }

    // Save the final iteration of this run
    save_binary(recon, output_path);
    return 0;
}