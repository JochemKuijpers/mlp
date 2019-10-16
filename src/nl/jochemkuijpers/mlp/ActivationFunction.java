package nl.jochemkuijpers.mlp;

public interface ActivationFunction {
    ActivationFunction LINEAR = new ActivationFunction() {
        @Override
        public float apply(float x) {
            return x;
        }

        @Override
        public float derivative(float x) {
            return 1;
        }
    };
    ActivationFunction RELU = new ActivationFunction() {
        @Override
        public float apply(float x) {
            return x > 0 ? x : 0;
        }

        @Override
        public float derivative(float x) {
            return x > 0 ? 1 : x == 0 ? 0.5f : 0;
        }
    };
    ActivationFunction LEAKY_RELU = new ActivationFunction() {
        private final static float alpha = 0.1f;

        @Override
        public float apply(float x) {
            return x > 0 ? x : x * alpha;
        }

        @Override
        public float derivative(float x) {
            return x > 0 ? 1 : x == 0 ? alpha * 0.5f + 0.5f : alpha;
        }
    };
    ActivationFunction SIGMOID = new ActivationFunction() {
        @Override
        public float apply(float x) {
            return 1 / (1 + (float) Math.exp(-x));
        }

        @Override
        public float derivative(float x) {
            float sigmoid = apply(x);
            return sigmoid * (1 - sigmoid);
        }
    };
    ActivationFunction TANH = new ActivationFunction() {
        @Override
        public float apply(float x) {

            float ex = (float) Math.exp(x);
            float enx = (float) Math.exp(-x);
            return (ex - enx) / (ex + enx);
        }

        @Override
        public float derivative(float x) {
            float tanh = apply(x);
            return 1 - tanh * tanh;
        }
    };

    float apply(float x);

    float derivative(float x);

    default float[] applyAll(float[] xs) {
        float[] ys = new float[xs.length];
        for (int i = 0; i < ys.length; i++) {
            ys[i] = apply(xs[i]);
        }
        return ys;
    }

    default float[] derivativeAll(float[] xs) {
        float[] ys = new float[xs.length];
        for (int i = 0; i < ys.length; i++) {
            ys[i] = derivative(xs[i]);
        }
        return ys;
    }
}
