from app.generators.sdv_generator import SDVGenerator


class TVAEGenerator(SDVGenerator):
    name = "tvae"
    synthesizer_class = "TVAESynthesizer"
    neural = True
