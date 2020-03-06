#include <cmath>

namespace YoloV2Tiny {
    std::vector<DetectedObject_t> run_nms(std::vector<DetectedObject_t> candidates, double threshold) {
        std::vector<DetectedObject_t> nms_candidates;
        std::sort(candidates.begin(), candidates.end());

        while (candidates.size() > 0) {
            auto p_first_candidate = candidates.begin();
            const auto &first_candidate = *p_first_candidate;
            double first_candidate_area = first_candidate.width * first_candidate.height;

            for (auto p_candidate = p_first_candidate + 1; p_candidate != candidates.end();) {
                const auto &candidate = *p_candidate;

                double inter_width = std::min(first_candidate.x + first_candidate.width, candidate.x + candidate.width) -
                                    std::max(first_candidate.x, candidate.x);
                double inter_height =
                    std::min(first_candidate.y + first_candidate.height, candidate.y + candidate.height) -
                    std::max(first_candidate.y, candidate.y);
                if (inter_width <= 0.0 || inter_height <= 0.0) {
                    ++p_candidate;
                    continue;
                }

                double inter_area = inter_width * inter_height;
                double candidate_area = candidate.width * candidate.height;

                double overlap = inter_area / std::min(candidate_area, first_candidate_area);
                if (overlap > threshold)
                    p_candidate = candidates.erase(p_candidate);
                else
                    ++p_candidate;
            }

            nms_candidates.push_back(first_candidate);
            candidates.erase(p_first_candidate);
        }

        return nms_candidates;
    }

    static void scaleBack(bool keep_ratio, float &xmin, float &xmax, float &ymin, float &ymax, int image_width,
                        int image_height) {
        if (!keep_ratio) {
            xmin *= image_width;
            xmax *= image_width;
            ymin *= image_height;
            ymax *= image_height;
        } else {
            // ratio between netw/neth is matter, not the absolute value
            int netw = 416;
            int neth = 416;
            float scalew = float(netw) / image_width;
            float scaleh = float(neth) / image_height;
            float scale = scalew < scaleh ? scalew : scaleh;
            float new_width = image_width * scale;
            float new_height = image_height * scale;
            float pad_w = (netw - new_width) / 2.0;
            float pad_h = (neth - new_height) / 2.0;

            xmin = (xmin * netw - pad_w) / scale;
            xmax = (xmax * netw - pad_w) / scale;
            ymin = (ymin * neth - pad_h) / scale;
            ymax = (ymax * neth - pad_h) / scale;
        }
        if (xmin < 0)
            xmin = 0;
        if (xmax > image_width)
            xmax = image_width;
        if (ymin < 0)
            ymin = 0;
        if (ymax > image_height)
            ymax = image_height;
    }

    const int num_classes = 20;
    enum RawNetOut {
        idxX = 0,
        idxY = 1,
        idxW = 2,
        idxH = 3,
        idxScale = 4,
        idxClassProbBegin = 5,
        idxClassProbEnd = idxClassProbBegin + num_classes,
        idxCount = idxClassProbEnd
    };

    // void fillRawNetOutMoviTL(float const *pIn, const int anchor_idx, const int cell_ind, const float threshold,
    //                         float *pOut) {

    //     const int strides4D[] = {13 * 128, 128, 25, 1};
    //     for (int l = 0; l < idxCount; l++) {
    //         const int ind = cell_ind * strides4D[1] + anchor_idx * strides4D[2] + l * strides4D[3];
    //         pOut[l] = dequantize((reinterpret_cast<const u_int8_t*>(pIn))[ind]);
    //     }
    //     pOut[idxX] = sigmoid(pOut[idxX]);                                       // x
    //     pOut[idxY] = sigmoid(pOut[idxY]);                                       // y
    //     pOut[idxScale] = sigmoid(pOut[idxScale]);                               // scale
    //     softMax(pOut + idxClassProbBegin, idxClassProbEnd - idxClassProbBegin); // probs
    //     for (int l = idxClassProbBegin; l < idxClassProbEnd; ++l) {
    //         pOut[l] *= pOut[idxScale]; // scaled probs
    //         if (pOut[l] <= threshold)
    //             pOut[l] = 0.f;
    //     }
    // }

    void fillRawNetOut(float const *pIn, const int anchor_idx, const int cell_ind, const float threshold,
                    float *pOut) {
        const int kAnchorSN = 13;
        const int kOutBlobItemN = 25;
        const int k2Depths = kAnchorSN * kAnchorSN;
        const int k3Depths = k2Depths * kOutBlobItemN;

        const int commonOffset = anchor_idx * k3Depths + cell_ind;
        pOut[idxX] = pIn[commonOffset + 1 * k2Depths];     // x
        pOut[idxY] = pIn[commonOffset + 0 * k2Depths];     // y
        pOut[idxW] = pIn[commonOffset + 3 * k2Depths];     // w
        pOut[idxH] = pIn[commonOffset + 2 * k2Depths];     // h
        pOut[idxScale] = pIn[commonOffset + 4 * k2Depths]; // scale
        for (int l = idxClassProbBegin; l < idxClassProbEnd; ++l) {
            pOut[l] = pIn[commonOffset + l * k2Depths] * pOut[idxScale]; // scaled probs
            if (pOut[l] <= threshold)
                pOut[l] = 0.f;
        }
    }

    using rawNetOutExtractor =
        std::function<void(float const *, const int, const int, const float threshold, float *)>;

    std::vector<DetectedObject_t> TensorToBBoxYoloV2TinyCommon(const InferenceEngine::Blob::Ptr &blob, int image_height, int image_width,
                                     double thresholdConf, rawNetOutExtractor extractor) {
        int kAnchorSN = 13;
        float kAnchorScales[] = {1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f};

        const float *data = (const float *)blob->buffer();
        if (data == nullptr) {
            printf("Blob data pointer is null");
        }

        const bool preprocess_keep_ratio = false;

        float raw_netout[idxCount];
        std::vector<DetectedObject_t> objects;
        for (int k = 0; k < 5; k++) {
            float anchor_w = kAnchorScales[k * 2];
            float anchor_h = kAnchorScales[k * 2 + 1];

            for (int i = 0; i < kAnchorSN; i++) {
                for (int j = 0; j < kAnchorSN; j++) {
                    extractor(data, k, i * kAnchorSN + j, thresholdConf, raw_netout);

                    std::pair<int, float> max_info = std::make_pair(0, 0.f);
                    for (int l = idxClassProbBegin; l < idxClassProbEnd; l++) {
                        float class_prob = raw_netout[l];
                        if (class_prob > 1.f) {
                            printf("class_prob weired %f", class_prob);
                        }
                        if (class_prob > max_info.second) {
                            max_info.first = l - idxClassProbBegin;
                            max_info.second = class_prob;
                        }
                    }

                    if (max_info.second > thresholdConf) {
                        // scale back to image width/height
                        float cx = (j + raw_netout[idxX]) / kAnchorSN;
                        float cy = (i + raw_netout[idxY]) / kAnchorSN;
                        float w = std::exp(raw_netout[idxW]) * anchor_w / kAnchorSN;
                        float h = std::exp(raw_netout[idxH]) * anchor_h / kAnchorSN;
                        float x0 = cx - w * 0.5f;
                        float y0 = cy - h * 0.5f;
                        float x1 = cx + w * 0.5f;
                        float y1 = cy + h * 0.5f;
                        scaleBack(preprocess_keep_ratio, x0, x1, y0, y1, image_width, image_height);

                        DetectedObject_t object((x0 + x1) * 0.5, (y0 + y1) * 0.5, y1 - y0, x1 - x0, max_info.second);
                        objects.push_back(object);
                    }
                }
            }
        }

        double nms_threshold = 0.5;
        objects = run_nms(objects, nms_threshold);

        // for (const DetectedObject_t &object : objects)
        // {
        //     printf("*****yolotiny @(%d, %d), WxH=%dx%dx, prob=%.1f", (object.x >= 0) ? object.x : 0,
        //         (object.y >= 0) ? object.y : 0, object.width, object.height, object.confidence);
        // }

        // TODO: add ROI meta params?

        return objects;
    }

} // namespace YoloV2Tiny
