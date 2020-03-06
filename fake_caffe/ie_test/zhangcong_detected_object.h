struct DetectedObject_t {
    int x;
    int y;
    int width;
    int height;
    float confidence;
    explicit DetectedObject_t(float x, float y, float h, float w, float confidence, float h_scale = 1.f,
                            float w_scale = 1.f)
        : x(static_cast<int>((x - w / 2) * w_scale)), y(static_cast<int>((y - h / 2) * h_scale)),
        width(static_cast<int>(w * w_scale)), height(static_cast<int>(h * h_scale)), confidence(confidence) {
    }
    DetectedObject_t() = default;
    ~DetectedObject_t() = default;
    DetectedObject_t(const DetectedObject_t &) = default;
    DetectedObject_t(DetectedObject_t &&) = default;
    DetectedObject_t &operator=(const DetectedObject_t &) = default;
    DetectedObject_t &operator=(DetectedObject_t &&) = default;
    bool operator<(const DetectedObject_t &other) const {
        return this->confidence > other.confidence; //TODO fix me
    }
};