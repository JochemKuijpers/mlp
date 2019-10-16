package nl.jochemkuijpers.mlp;

public interface ErrorFunction {
    ErrorFunction MEAN_SQUARE_ERROR = new ErrorFunction() {
        @Override
        public float apply(float[] zs, float[] ys) {
            assert (zs.length == ys.length);
            float sum = 0;
            for (int i = 0; i < zs.length; i++) {
                sum += (zs[i] - ys[i]) * (zs[i] - ys[i]);
            }
            return sum / zs.length;
        }

        @Override
        public float[] derivative(float[] zs, float[] ys) {
            assert (zs.length == ys.length);
            float[] ds = new float[zs.length];
            for (int i = 0; i < ds.length; i++) {
                ds[i] = 2 * (zs[i] - ys[i]);
            }
            return ds;
        }
    };

    float apply(float[] zs, float[] ys);

    float[] derivative(float[] zs, float[] ys);
}
