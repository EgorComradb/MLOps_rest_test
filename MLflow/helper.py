class Helper:
    __model_info = None
    __input_example = None

    def set_model_info(self, model_info):
        self.__model_info = model_info

    def get_model_info(self):
        return self.__model_info

    def set_input_example(self, input_example):
        self.__input_example = input_example

    def get_input_example(self):
        return self.__input_example