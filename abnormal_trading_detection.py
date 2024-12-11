import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import xgboost as xgb  # XGBoost Classifier
import pickle
from io import StringIO  # Add this line at the top of your script
from io import BytesIO

pro_data = pd.read_excel('/content/drive/My Drive/P2 code/process_data.xlsx',dtype={'Stock_Code': str})
fx_data=pd.read_excel('/content/drive/My Drive/P2 code/data_results.xlsx',dtype={'Stock_Code': str})

filename = {
    "lr": "log_model.pkl",
    "rf": "rf_model.pkl",
    "xgb": "xgb_model.pkl"
}

# Load the model
def load_logistic_regression():
    return pickle.load(open(filename["lr"], "rb"))

def load_random_forest():
    return pickle.load(open(filename["rf"], "rb"))

def load_xgb():
    return pickle.load(open(filename["xgb"], "rb"))

def generate_input_from_excel():
    df = pd.read_excel('/content/drive/My Drive/P2 code/data_results.xlsx').drop(['anomaly', 'Date', 'Stock_Code', 'Stock_Name','Short_Selling_Sold_Volume_10K_Shares','Short_Selling_Repurchased_Volume_10K_Shares',
                    'Short_Selling_Volume_10K_Shares','Margin_Purchase_Amount_10K_Yuan','Margin_Repurchased_Amount_10K_Yuan','Closing_Price_Yuan'], axis=1)
    return df.sample().reset_index(drop=True)

def generate_model_result(x=None):
    lr_model = load_logistic_regression()
    rf_model = load_random_forest()
    xgb_model = load_xgb()

    x = x if x is not None else generate_input_from_excel()

    lr_result = lr_model.predict(x)[0]
    rf_result = rf_model.predict(x)[0]
    xgb_result = xgb_model.predict(x)[0]
    
    return {"input": x, "lr_result": lr_result, "rf_result": rf_result, "xgb_result":xgb_result}

# 3-sigma 异常检测逻辑
def detect_anomalies(data):
    results = []  # 保存结果
    data_cols = data.columns.tolist()  # 获取所有列名
    threshold_manually = {
        "Closing_Price_Yuan": "3-sigma",
        "Price_Change_Percent": "3-sigma",
        "Margin_Balance_10K_Yuan": "3-sigma",
        "Balance_to_Circulating_MV_Percent": "3-sigma",
        "Margin_Purchase_Amount_10K_Yuan": "3-sigma",
        "Margin_Repurchased_Amount_10K_Yuan": "3-sigma",
        "Margin_Net_Purchase_Amount_10K_Yuan": "3-sigma",
        "Short_Selling_Balance_10K_Yuan": "3-sigma",
        "Short_Selling_Volume_10K_Shares": "3-sigma",
        "Short_Selling_Sold_Volume_10K_Shares": "3-sigma",
        "Short_Selling_Repurchased_Volume_10K_Shares": "3-sigma",
        "Short_Selling_Net_Sold_Volume_10K_Shares": "3-sigma",
        "Margin_and_Short_Balance_10K_Yuan": "3-sigma",
        "Margin_and_Short_Balance_Change_10K_Yuan": "3-sigma",
    }
    all_coda_data = data.groupby(by="Stock_Code")  # 按证券代码分组
    for coda_id, code_data in all_coda_data:
        code_data = code_data.sort_values(by="Date", ascending=True).reset_index(drop=True)  # 按日期排序
        # 计算阈值
        threshold = {}
        for col in threshold_manually.keys():
            if threshold_manually[col] == "3-sigma":
                min_value = data[col].mean() - 3 * data[col].std()
                max_value = data[col].mean() + 3 * data[col].std()
                threshold[col] = (min_value, max_value)

        # 标记异常值
        for col in threshold.keys():
            code_data[f"{col}_anomaly"] = code_data[col].apply(
                lambda x: 0 if threshold[col][0] <= x <= threshold[col][1] else 1
            )

        # 汇总异常标签
        code_data["anomaly"] = code_data[[col for col in code_data.columns if "anomaly" in col]].sum(axis=1)
        code_data["anomaly"] = code_data["anomaly"].apply(lambda x: 1 if x > 0 else 0)

        results.append(code_data[data_cols + ["anomaly"]])  # 保存结果

    # 合并所有结果
    results = pd.concat(results, axis=0)
    results = results.sort_values(by=["Stock_Code", "Date"], ascending=True).reset_index(drop=True)
    return results

def predict_note_authentication(Price_Change_Percent,Margin_Balance_10K_Yuan,Balance_to_Circulating_MV_Percent,Margin_Net_Purchase_Amount_10K_Yuan,Short_Selling_Balance_10K_Yuan,
                                Short_Selling_Net_Sold_Volume_10K_Shares,Margin_and_Short_Balance_10K_Yuan,Margin_and_Short_Balance_Change_10K_Yuan):
    
    inputs = [Price_Change_Percent,Margin_Balance_10K_Yuan,Balance_to_Circulating_MV_Percent,Margin_Net_Purchase_Amount_10K_Yuan,Short_Selling_Balance_10K_Yuan,
                                Short_Selling_Net_Sold_Volume_10K_Shares,Margin_and_Short_Balance_10K_Yuan,Margin_and_Short_Balance_Change_10K_Yuan]
    rf_model = load_random_forest()
    prediction=rf_model.predict([inputs])
    print(prediction)
    return prediction

# 提取唯一的映射关系
unique_mapping = fx_data[["Stock_Name", "Stock_Code"]].drop_duplicates()
# 生成反向映射字典
reverse_mapping = dict(zip(unique_mapping["Stock_Code"], unique_mapping["Stock_Name"]))


# https://icons.getbootstrap.com/ for icons

def streamlit_menu(example=1, options=["Home", "Contact"], icons=["coin", "bar-chart"]):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=options,  # required
                icons=icons,  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=options,  # required
            icons=icons,  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 3. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=options,  # required
            icons=icons,  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected
    


# Streamlit UI
# Set page configuration
st.set_page_config(layout="wide")

# 1 = sidebar menu, 2 = horizontal menu, 3 = horizontal menu w/ custom menu
selected = streamlit_menu(example = 1, 
                          options=["About","Detection", "EDA", "Prediction"],
                          icons=["house", "bar-chart-steps","bar-chart-fill",  "file-earmark-medical-fill"])



#**************************
if selected == "About":
    # Title of the page
    st.markdown("<h2>Detection of Anomalous Trading Behaviors in China's Securities Margin Trading Systems</h2>", unsafe_allow_html=True)
    st.image("https://github.com/wangyouqing0416/final-project_wyq/blob/main/first%20page.jpg", caption="What's the margin trading ?", use_column_width=True)
    st.markdown("[click here to jump to the source data URL](https://data.eastmoney.com/rzrq/detail/all.1.2024-01-02.html)")

#**************************
if selected == "Detection":
    uploaded_file = st.file_uploader("请上传 Excel 文件，对异常情况进行检测", type=["xlsx"])
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("###Uploaded data：")
        st.dataframe(data)

        # 检测按钮
        if st.button("Start detect"):
            with st.spinner("Detecting anomalies..."):
                results = detect_anomalies(pro_data)  # 应用检测逻辑

            # 显示结果数据框
            st.success("Detection completed!")
            st.dataframe(results)

            # 提供下载按钮
            def to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False, sheet_name="Results")
                return output.getvalue()

            excel_data = to_excel(results)

            st.download_button(
                label="Download Results",
                data=excel_data,
                file_name="anomaly_detection_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    

#***************************************
#可以实现输入证券代码，显示它这个月的时间序列图，将异常的时间点标记
if selected == "EDA":
    # Title of the page
    st.markdown("<h2>Visualisation Dashboard</h2>", unsafe_allow_html=True)
    # 选择证券代码
    stock_code = st.selectbox("please select stock code：", fx_data["Stock_Code"].unique())

    # 确保日期列为 datetime 类型
    pro_data["Date"] = pd.to_datetime(pro_data["Date"])

    # 获取时间范围并转换为 datetime 对象
    min_date = pro_data["Date"].min().to_pydatetime()
    max_date = pro_data["Date"].max().to_pydatetime()

    # 选择时间范围
    date_range = st.slider(
        "选择时间范围：",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # 筛选数据
    filtered_data = pro_data[(pro_data["Stock_Code"] == stock_code) & 
                             (pro_data["Date"] >= date_range[0]) & 
                             (pro_data["Date"] <= date_range[1])]

    # 检测异常值
    with st.spinner("计算异常点..."):
        filtered_data_with_anomalies = detect_anomalies(filtered_data)

    # 只显示标记为异常的数据
    anomalies = filtered_data_with_anomalies[filtered_data_with_anomalies["anomaly"] == 1]

    if not anomalies.empty:
        st.write("### 异常点时间序列")

        # 绘制时间序列图
        fig = px.line(
            filtered_data_with_anomalies,
            x="Date",
            y="anomaly",
            title=f"{reverse_mapping[stock_code]} ({stock_code}) 的异常点时间序列",
            markers=True,
        )

        # 标记异常点
        fig.add_scatter(
            x=anomalies["Date"],
            y=anomalies["anomaly"],
            mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            name="异常点"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示异常点数据表
        st.write("### 异常点数据")
        st.dataframe(anomalies)
    else:
        st.warning("在所选范围内没有异常点！")



#***************************************
if selected == "Prediction":
    trade_date=input_method = st.radio("Select Input Method", ('Manual Input', 'Random input'))
    if input_method == 'Manual Input':
        st.date_input("please select date",value=pd.Timestamp("2024-01-01") )
        col1, col2 = st.columns(2)
        with col1:
            stock_code=st.selectbox("stock code", list(reverse_mapping.keys()))
            stock_name=st.text_input("Stock Name", value=reverse_mapping[stock_code])
            Closing_Price_Yuan=st.number_input("closing price",min_value=1)
            Price_Change_Percent=st.number_input("Price Change Percent")
            Margin_Balance_10K_Yuan=st.number_input("Margin Balance 10K Yuan")
            Balance_to_Circulating_MV_Percent=st.number_input("Balance to Circulating MV Percent")
            Margin_Purchase_Amount_10K_Yuan=st.number_input("Margin Purchase Amount 10K Yuan")
            Margin_Repurchase_Amount_10K_Yuan=st.number_input("Margin Repurchase Amount 10K Yuan")
        with col2:
            Margin_Net_Purchase_Amount_10K_Yuan=st.number_input("Margin Net Purchase Amount 10K Yuan")
            Short_Selling_Balance_10K_Yuan=st.number_input("Short Selling Balance 10K Yuan")
            Short_Selling_Volume_10K_Shares=st.number_input("Short Selling Volume 10K Shares")
            Short_Selling_Sold_Volume_10K_Shares=st.number_input("Short Selling Sold Volume 10K Shares")
            Short_Selling_Repurchased_Volume_10K_Shares=st.number_input("Short Selling Repurchased Volume 10K Shares")
            Short_Selling_Net_Sold_Volume_10K_Shares=st.number_input("Short Selling Net Sold Volume 10K Shares")
            Margin_and_Short_Balance_10K_Yuan=st.number_input("Margin and Short Balance 10K Yuan")
            Margin_and_Short_Balance_Change_10K_Yuan=st.number_input("Margin and Short Balance Change 10K Yuan")
        if st.button("Predict"):
            result=predict_note_authentication(Price_Change_Percent,Margin_Balance_10K_Yuan,Balance_to_Circulating_MV_Percent,Margin_Net_Purchase_Amount_10K_Yuan,Short_Selling_Balance_10K_Yuan,
                                Short_Selling_Net_Sold_Volume_10K_Shares,Margin_and_Short_Balance_10K_Yuan,Margin_and_Short_Balance_Change_10K_Yuan)
            if result == 1:
                st.error('Alert! This is an abnormal data!', icon="❗")
            else:
                st.success('This is a normal data', icon="✅")

    elif input_method == 'Random input':
        st.title("Anomalous Trading Detection")
        st.text("This website is to show case the model trained for Anomalous Trading detection.")

        # Button Workflow
        st.subheader("1. Generate Random Input")

        if 'stage' not in st.session_state:
            st.session_state.stage = 0

        def set_state(i, x=None):
            st.session_state.stage = i

        if st.session_state.stage == 0:
            st.button('Generate', on_click=set_state, args=[1])

        # 1. Generate Random Input
        if st.session_state.stage >= 1:
            if st.session_state.stage == 1:
                x = generate_input_from_excel()
                st.session_state.x = x
    
            x = st.session_state.x
            st.dataframe(x)
            st.button('Generate', on_click=set_state, args=[1])

            st.subheader("2. Detect whether is Fraud")

            st.button('Detect', on_click=set_state, args=[2])

        # 2. Inference
        if st.session_state.stage >= 2:
            x = st.session_state.x
            result = generate_model_result(x)

            col1, col2, col3 = st.columns(3)

            with col1:
                with st.container(height=100):
                    st.subheader("Logistic Regression")
        
                if result["lr_result"]:
                    st.error('Alert! This is an abnormal data!', icon="❗")
                else:
                    st.success('This is a normal data', icon="✅")

            with col2:
                with st.container(height=100):
                    st.subheader("Random Forest")
        
                if result["rf_result"]:
                    st.error('Alert! This is an abnormal data!', icon="❗")
                else:
                    st.success('This is a normal data', icon="✅")

            with col3:
                with st.container(height=100):
                    st.subheader("XGB")

                if result["xgb_result"]:
                    st.error('Alert! This is an abnormal data!', icon="❗")
                else:
                    st.success('This is a normal data', icon="✅")
    
            st.subheader("3. Start Over")
            if st.session_state.stage == 2:
                st.button('Start Over', on_click=set_state, args=[0])

#               Closing_Price_Yuan,Price_Change_Percent,Margin_Balance_10K_Yuan,
#                    Balance_to_Circulating_MV_Percent,Margin_Purchase_Amount_10K_Yuan,Margin_Repurchase_Amount_10K_Yuan,Margin_Net_Purchase_Amount_10K_Yuan,
#                   Short_Selling_Balance_10K_Yuan,Short_Selling_Volume_10K_Shares,Short_Selling_Sold_Volume_10K_Shares,
#                   Short_Selling_Repurchased_Volume_10K_Shares,Short_Selling_Net_Sold_Volume_10K_Shares,Margin_and_Short_Balance_10K_Yuan,Margin_and_Short_Balance_Change_10K_Yuan