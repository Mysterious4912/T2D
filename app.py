from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import xgboost as xgb
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# 加载模型
model = xgb.Booster()
model.load_model('T2D_model.json')

# 加载配置文件
def load_config():
    with open('model_config.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config['feature_order'], config['default_values']

feature_order, default_values = load_config()

def load_threshold():
    with open('threshold_final.json', 'r', encoding='utf-8') as threshold_file:
        threshold_data = json.load(threshold_file)
    return threshold_data['threshold']

threshold_data = load_threshold()

# 首页路由
@app.route('/')
def home():
    return render_template('T2D.html')


# 配置加载路由（前端默认值）
@app.route('/config')
def get_config():
    return jsonify({'default_values': default_values})


# 预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取用户输入的数据
        data = request.get_json()
        print("📌 收到的数据:", data)

        # 补全默认值（从numeric字典获取）
        final_data = []
        for feature in feature_order:
            value = data.get(feature)
            if value not in (None, ""):  # 处理空字符串或缺失值
                final_data.append(float(value))  # 确保数值类型
            else:
                final_data.append(default_values["numeric"].get(feature, 0))

        print("✅ 最终输入数据:", final_data)

        # 创建DataFrame时指定列名
        input_df = pd.DataFrame([final_data], columns=feature_order)
        dtest = xgb.DMatrix(input_df)  # DMatrix自动使用DataFrame的列名

        # 进行模型预测
        prediction = model.predict(dtest)[0]
        print("🎯 预测概率:", prediction)

        # 判断风险等级
        risk_level = "高风险" if prediction > threshold_data else "低风险"

        # 返回结果
        return jsonify({
            "risk_level": risk_level,
            "probability": float(prediction)
        })

    except Exception as e:
        print("⚠️ 预测异常:", str(e))
        return jsonify({"error": f"预测失败: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
