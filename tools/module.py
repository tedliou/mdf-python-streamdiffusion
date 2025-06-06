import json
import os

class Config:
    def __init__(self, path="config.json"):
        self.path = path
        self.data = {
            "name": "",
            "version": "",
            "inputs": [],
            "input_params": {},
            "process": [],
            "process_params": {},
            "outputs": [],
            "output_params":{}
        }
        self.is_new_file = self.load()
    
    # 載入設定檔
    def load(self):
        file_exist = os.path.exists(self.path)
        is_new_file = True
        cfg = None
        
        # 如檔案存在，嘗試讀取內容
        if file_exist:
            try:
                with open(self.path, "r", encoding="utf-8") as file:
                    cfg = json.load(file)

                self.data = cfg
                is_new_file = False

            except json.JSONDecodeError as e:
                # 讀取時發生 JSON 剖析失敗，設定 is_new_file 為 True 以建立預設參數
                print(e.msg)
                is_new_file = True

            except Exception as e:
                # 讀取時發生未知錯誤，，設定 is_new_file 為 True 以建立預設參數
                print(e)
                is_new_file = True
        
        # 使用預設參數並存檔
        if is_new_file:
            self.save()

        return is_new_file
    
    
    # 儲存設定檔
    def save(self):
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)
    
    # 設定模組名稱
    def set_name(self, module_name):
        self.data["name"] = module_name
        self.save()
    
    # 設定模組版本
    def set_version(self, version_value):
        self.data["version"] = version_value
        self.save()
    

    # 設定輸入方法
    def set_inputs(self, param_value):
        self.data["inputs"] = param_value
        self.save()

    # 取得輸入方法
    def get_inputs(self):
        return self.data["inputs"]
    

    # 設定輸入參數
    def set_input_params(self, param_name, param_value):
        self.data["input_params"][param_name] = param_value
        self.save()
    
    # 取得輸入參數
    def get_input_params(self, param_name):
        return self.data["input_params"][param_name]
    

    # 設定輸出方法
    def set_outputs(self, param_value):
        self.data["outputs"] = param_value
        self.save()

    # 取得輸出方法
    def get_outputs(self):
        return self.data["outputs"]
    

    # 設定輸出參數
    def set_output_params(self, param_name, param_value):
        self.data["output_params"][param_name] = param_value
        self.save()
    
    # 取得輸出參數
    def get_output_params(self, param_name):
        return self.data["output_params"][param_name]
    

    # 設定處理方法
    def set_process(self, param_value):
        self.data["process"] = param_value
        self.save()

    # 取得處理方法
    def get_process(self):
        return self.data["process"]
    

    # 設定處理參數
    def set_process_params(self, param_name, param_value):
        self.data["process_params"][param_name] = param_value
        self.save()
    
    # 取得輸出參數
    def get_process_params(self, param_name):
        return self.data["process_params"][param_name]
