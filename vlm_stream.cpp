// vlm_stream.cpp - Persistent VLM binary for live stream inference
// Keeps models loaded, accepts new images via stdin
// Protocol: read "IMAGE_PATH\tPROMPT\n", output between delimiters
// Send "EXIT\n" to quit

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <csignal>
#include <opencv2/opencv.hpp>
#include "image_enc.h"
#include "rkllm.h"

using namespace std;

LLMHandle llmHandle = nullptr;
rknn_app_context_t rknn_app_ctx;
string response_buffer;
volatile bool inference_done = false;

void exit_handler(int signal) {
    if (llmHandle != nullptr) {
        LLMHandle tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(tmp);
    }
    release_imgenc(&rknn_app_ctx);
    exit(signal);
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_FINISH) {
        inference_done = true;
    } else if (state == RKLLM_RUN_ERROR) {
        fprintf(stderr, "inference error\n");
        inference_done = true;
    } else if (state == RKLLM_RUN_NORMAL) {
        response_buffer += result->text;
    }
    return 0;
}

cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& bg) {
    int w = img.cols, h = img.rows;
    if (w == h) return img.clone();
    int size = max(w, h);
    cv::Mat result(size, size, img.type(), bg);
    img.copyTo(result(cv::Rect((size - w) / 2, (size - h) / 2, w, h)));
    return result;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s encoder_model llm_model max_new_tokens max_context_len [core_num]\n", argv[0]);
        return 1;
    }

    const char* encoder_path = argv[1];
    const char* llm_path = argv[2];
    int max_new_tokens = atoi(argv[3]);
    int max_context_len = atoi(argv[4]);
    int core_num = argc > 5 ? atoi(argv[5]) : 3;

    signal(SIGINT, exit_handler);
    signal(SIGTERM, exit_handler);

    // Init LLM
    fprintf(stderr, "Loading LLM model...\n");
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = llm_path;
    param.top_k = 1;
    param.top_p = 0.9;
    param.temperature = 0.8;
    param.repeat_penalty = 1.1;
    param.frequency_penalty = 0.0;
    param.presence_penalty = 0.0;
    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;
    param.extend_param.embed_flash = 1;
    param.img_start = "<|vision_start|>";
    param.img_end = "<|vision_end|>";
    param.img_content = "<|image_pad|>";

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret != 0) {
        fprintf(stderr, "rkllm init failed\n");
        return 1;
    }
    fprintf(stderr, "LLM loaded OK\n");

    // Init image encoder
    fprintf(stderr, "Loading vision encoder...\n");
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    ret = init_imgenc(encoder_path, &rknn_app_ctx, core_num);
    if (ret != 0) {
        fprintf(stderr, "init_imgenc failed\n");
        return 1;
    }
    fprintf(stderr, "Vision encoder loaded OK\n");

    size_t n_image_tokens = rknn_app_ctx.model_image_token;
    size_t embed_size = rknn_app_ctx.model_embed_size;
    size_t n_outputs = rknn_app_ctx.io_num.n_output;
    size_t image_width = rknn_app_ctx.model_width;
    size_t image_height = rknn_app_ctx.model_height;

    int embed_len = n_image_tokens * embed_size * n_outputs;
    float* img_vec = new float[embed_len];

    RKLLMInput rkllm_input;
    RKLLMInferParam infer_params;
    memset(&infer_params, 0, sizeof(RKLLMInferParam));
    infer_params.mode = RKLLM_INFER_GENERATE;
    infer_params.keep_history = 0;

    // Signal ready
    printf("---READY---\n");
    fflush(stdout);

    string line;
    while (getline(cin, line)) {
        if (line == "EXIT") break;

        // Parse: IMAGE_PATH\tPROMPT
        size_t tab = line.find('\t');
        if (tab == string::npos) {
            printf("---ERROR---\nInvalid input format. Use: IMAGE_PATH\\tPROMPT\n---END---\n");
            fflush(stdout);
            continue;
        }

        string image_path = line.substr(0, tab);
        string prompt = "<image>" + line.substr(tab + 1);

        // Load and preprocess image
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            printf("---ERROR---\nFailed to load image: %s\n---END---\n", image_path.c_str());
            fflush(stdout);
            continue;
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat square = expand2square(img, cv::Scalar(127.5, 127.5, 127.5));
        cv::Mat resized;
        cv::resize(square, resized, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);

        // Encode image
        memset(img_vec, 0, embed_len * sizeof(float));
        ret = run_imgenc(&rknn_app_ctx, resized.data, img_vec);
        if (ret != 0) {
            printf("---ERROR---\nImage encoding failed\n---END---\n");
            fflush(stdout);
            continue;
        }

        // Run LLM inference
        response_buffer.clear();
        inference_done = false;

        memset(&rkllm_input, 0, sizeof(RKLLMInput));
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.role = "user";
        rkllm_input.multimodal_input.prompt = (char*)prompt.c_str();
        rkllm_input.multimodal_input.image_embed = img_vec;
        rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
        rkllm_input.multimodal_input.n_image = 1;
        rkllm_input.multimodal_input.image_height = image_height;
        rkllm_input.multimodal_input.image_width = image_width;

        printf("---RESPONSE_START---\n");
        fflush(stdout);

        rkllm_run(llmHandle, &rkllm_input, &infer_params, NULL);

        printf("%s\n---RESPONSE_END---\n", response_buffer.c_str());
        fflush(stdout);

        // Clear KV cache for next independent query
        rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
    }

    delete[] img_vec;
    release_imgenc(&rknn_app_ctx);
    rkllm_destroy(llmHandle);
    return 0;
}
