export void Add(uniform float out[], uniform const float in1[],
                uniform const float in2[], uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] + in2[index];
  }
}

export void Sub(uniform float out[], uniform const float in1[],
                uniform const float in2[], uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] - in2[index];
  }
}

export void Mul(uniform float out[], uniform const float in1[],
                uniform const float in2[], uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] * in2[index];
  }
}

export void Div(uniform float out[], uniform const float in1[],
                uniform const float in2[], uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] / in2[index];
  }
}

export void AddConst(uniform float out[], uniform const float in1[],
                     uniform const float in2, uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] + in2;
  }
}

export void SubConst(uniform float out[], uniform const float in1[],
                     uniform const float in2, uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] - in2;
  }
}

export void MulConst(uniform float out[], uniform const float in1[],
                     uniform const float in2, uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] * in2;
  }
}

export void DivConst(uniform float out[], uniform const float in1[],
                     uniform const float in2, uniform int count) {
  foreach (index = 0 ... count) {
    out[index] = in1[index] / in2;
  }
}