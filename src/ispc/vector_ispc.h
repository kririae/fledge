namespace ispc {
extern "C" {
extern void Add(float *out, const float *in1, const float *in2, int count);
extern void Sub(float *out, const float *in1, const float *in2, int count);
extern void Mul(float *out, const float *in1, const float *in2, int count);
extern void Div(float *out, const float *in1, const float *in2, int count);
extern void AddConst(float *out, const float *in1, const float in2, int count);
extern void SubConst(float *out, const float *in1, const float in2, int count);
extern void MulConst(float *out, const float *in1, const float in2, int count);
extern void DivConst(float *out, const float *in1, const float in2, int count);
}  // extern "C"
}  // namespace ispc
