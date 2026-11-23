from app.generators.sdv_generator import SDVGenerator


class GaussianCopulaGenerator(SDVGenerator):
    name = "gaussian_copula"
    synthesizer_class = "GaussianCopulaSynthesizer"
    neural = False
