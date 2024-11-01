import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우, 맑은 고딕 폰트를 사용합니다.
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Streamlit 웹앱 시작
st.title('평균기온으로 공급량 예측하기')

# 1번: 모델 생성 및 학습 파트
st.header('1. 모델 생성 및 학습')

# 모델 선택 옵션 박스
model_option = st.selectbox('모델을 선택하세요', ('다항회귀모델', '랜덤포레스트 모델'))

# 학습 데이터 기간 선택
train_start_year = st.selectbox('학습 데이터 시작 연도', list(range(2013, 2025)))
train_end_year = st.selectbox('학습 데이터 종료 연도', list(range(train_start_year, 2025)))

# 학습 버튼 클릭
if 'model' not in st.session_state:
    st.session_state.model = None
if 'poly' not in st.session_state:
    st.session_state.poly = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None

if st.button('학습 시작'):
    # 임의의 데이터를 생성합니다 (연도, 일별 평균 기온 및 공급량 데이터)
    np.random.seed(42)
    years = np.arange(2013, 2025)
    days_per_year = 365
    data = []
    for year in years:
        for day in range(1, days_per_year + 1):
            average_temp = np.random.uniform(-10, 35)
            supply = 100 + 5 * average_temp + np.random.normal(0, 10)  # 공급량은 평균기온과의 관계를 반영하여 생성
            data.append([year, day, average_temp, supply])

    data = pd.DataFrame(data, columns=['year', 'day_of_year', 'average_temp', 'supply'])

    # 선택된 학습 데이터 기간에 맞게 데이터 필터링
    st.session_state.train_data = data[(data['year'] >= train_start_year) & (data['year'] <= train_end_year)]

    X = st.session_state.train_data[['average_temp']]
    y = st.session_state.train_data['supply']

    # 모델 생성 및 학습
    if model_option == '다항회귀모델':
        st.session_state.poly = PolynomialFeatures(degree=3)
        X_poly = st.session_state.poly.fit_transform(X)
        st.session_state.model = LinearRegression()
        st.session_state.model.fit(X_poly, y)
        st.success('다항회귀모델 학습이 완료되었습니다.')
    elif model_option == '랜덤포레스트 모델':
        st.session_state.model = RandomForestRegressor(n_estimators=100, random_state=42)
        st.session_state.model.fit(X, y)
        st.success('랜덤포레스트 모델 학습이 완료되었습니다.')

# 2번: 예측하기
st.header('2. 예측하기')

# 날짜별 평균기온 입력 배열 박스
input_dates = st.text_area('날짜별 평균기온을 입력하세요 (예: 15.3, 18.7, 20.0)', '')

# 예측하기 버튼 클릭
if st.button('예측하기'):
    if input_dates:
        try:
            input_temps = list(map(float, input_dates.split(',')))
            X_input = pd.DataFrame({'average_temp': input_temps})

            # 예측 수행
            predictions = None
            if model_option == '다항회귀모델' and st.session_state.poly is not None:
                X_input_poly = st.session_state.poly.transform(X_input)
                predictions = st.session_state.model.predict(X_input_poly)
            elif model_option == '랜덤포레스트 모델' and st.session_state.model is not None:
                predictions = st.session_state.model.predict(X_input)

            if predictions is not None:
                # 예측 결과를 표 형태로 보여줌
                result_df = pd.DataFrame({'평균기온': input_temps, '예측공급량': predictions})
                st.write('예측 결과:')
                st.table(result_df.style.set_properties(**{'background-color': '#f0f0f0', 'color': '#333', 'border-color': '#ccc'}))

                # CSV 파일로 저장 및 다운로드 버튼 추가
                csv = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label='CSV 파일 다운로드',
                    data=csv,
                    file_name='예측결과.csv',
                    mime='text/csv'
                )

                # 예측 결과 시각화
                fig, ax = plt.subplots()
                # 기존 학습 데이터 시각화 (회색, 투명도 적용)
                ax.scatter(st.session_state.train_data['average_temp'], st.session_state.train_data['supply'], color='gray', alpha=0.5, label='기존 데이터')
                # 새로운 예측 데이터 시각화 (빨간색, 포인트 큼직하게 표시)
                ax.scatter(input_temps, predictions, color='red', marker='o', s=100, edgecolors='black', label='예측 데이터')
                ax.set_xlabel('평균기온')
                ax.set_ylabel('예측된 공급량')
                ax.set_title('평균기온에 따른 공급량 예측')
                ax.legend()
                st.pyplot(fig)
            else:
                st.error('모델이 학습되지 않았습니다. 먼저 학습을 진행해주세요.')
        except ValueError:
            st.error('입력 값이 올바르지 않습니다. 숫자를 콤마로 구분하여 입력해주세요.')
    else:
        st.warning('평균기온을 입력해주세요.')
