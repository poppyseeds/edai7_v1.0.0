from app.generators.sdv_generator import SDVGenerator


class CTGANGenerator(SDVGenerator):
    name = "ctgan"
    synthesizer_class = "CTGANSynthesizer"
    neural = True
