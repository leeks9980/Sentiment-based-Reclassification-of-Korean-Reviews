import streamlit as st
import pandas as pd
import os

# ---------------------------------------------------------
# 1. 사용자 설정 구역: 게임별 이미지를 여기서 지정하세요.
# ---------------------------------------------------------
GAME_IMAGES = {
    "배틀 그라운드": "https://cdn.akamai.steamstatic.com/steam/apps/578080/header.jpg",
    "PLAYERUNKNOWN'S BATTLEGROUNDS": "https://cdn.akamai.steamstatic.com/steam/apps/578080/header.jpg",
    "PUBG: BATTLEGROUNDS": "https://cdn.akamai.steamstatic.com/steam/apps/578080/header.jpg",
    "엘든 링": "https://cdn.akamai.steamstatic.com/steam/apps/1245620/header.jpg",
    "ELDEN RING": "https://cdn.akamai.steamstatic.com/steam/apps/1245620/header.jpg",
    "사이버펑크 2077": "https://cdn.akamai.steamstatic.com/steam/apps/1091500/header.jpg",
    "Cyberpunk 2077": "https://cdn.akamai.steamstatic.com/steam/apps/1091500/header.jpg",
    "다크 소울3": "https://cdn.akamai.steamstatic.com/steam/apps/374320/header.jpg",
    "DARK SOULS™ III": "https://cdn.akamai.steamstatic.com/steam/apps/374320/header.jpg",
    "Dark Souls III": "https://cdn.akamai.steamstatic.com/steam/apps/374320/header.jpg",
    "기본 이미지": "https://via.placeholder.com/800x400.png?text=No+Image"
}

@st.cache_data
def load_data():
    try:
        return pd.read_csv(r'D:\code\steam_reviews_analyzed.csv')
    except Exception as e:
        return None

df = load_data()

if df is None:
    st.error("❌ 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()

# [수정됨] 여기서 변수를 미리 정의해야 아래 코드들에서 사용할 수 있습니다.
unique_games = df['game_id'].unique()

if 'view_history' not in st.session_state:
    st.session_state.view_history = []
if 'selected_game_id' not in st.session_state:
    st.session_state.selected_game_id = None

def get_review_summary(positive_ratio):
    if positive_ratio >= 0.95: return "압도적 긍정적", "blue"
    elif positive_ratio >= 0.80: return "매우 긍정적", "blue"
    elif positive_ratio >= 0.70: return "긍정적", "green"
    elif positive_ratio >= 0.40: return "복합적", "orange"
    elif positive_ratio >= 0.20: return "부정적", "red"
    else: return "압도적 부정적", "red"

def show_game_details(game_name):
    st.session_state.selected_game_id = game_name
    if game_name not in st.session_state.view_history:
        st.session_state.view_history.append(game_name)

# ---------------------------------------------------------
# UI 구성: 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.title("🔍 메뉴")
    
    if st.session_state.selected_game_id:
        st.info(f"현재 선택됨:\n**{st.session_state.selected_game_id}**")
        if st.button("⬅️ 목록으로 돌아가기", type="primary", use_container_width=True):
            st.session_state.selected_game_id = None
            st.rerun()
        st.divider()

    search_query = st.text_input("게임 검색", placeholder="게임 이름을 입력하세요")

    st.subheader("최근 본 게임")
    if st.session_state.view_history:
        for game in reversed(st.session_state.view_history[-5:]):
            if st.button(f"🕒 {game}", key=f"history_{game}"):
                st.session_state.selected_game_id = game
                st.rerun()
    else:
        st.caption("기록 없음")

# ---------------------------------------------------------
# UI 구성: 메인 화면
# ---------------------------------------------------------
st.title("🎮 스팀 게임 리뷰 분석기")

# --- 상세 페이지 ---
if st.session_state.selected_game_id:
    game_name = st.session_state.selected_game_id
    
    # 상단 뒤로가기 버튼
    if st.button("⬅️ 뒤로 가기 (Back)", key="top_back"):
        st.session_state.selected_game_id = None
        st.rerun()

    img_url = GAME_IMAGES.get(game_name, GAME_IMAGES["기본 이미지"])
    st.image(img_url, use_container_width=True)
    
    if img_url == GAME_IMAGES["기본 이미지"]:
        st.warning(f"⚠️ 이미지가 없습니다! CSV 실제 이름: **'{game_name}'** (이 이름을 코드 상단에 추가하세요)")

    st.header(game_name)

    game_reviews = df[df['game_id'] == game_name]
    total_reviews = len(game_reviews)
    positive_count = game_reviews['corrected_voted_up'].sum()
    
    if total_reviews > 0:
        positive_ratio = positive_count / total_reviews
        positive_percent = int(positive_ratio * 100)
        negative_percent = 100 - positive_percent
    else:
        positive_ratio = 0
        positive_percent = 0
        negative_percent = 0

    summary_text, color_code = get_review_summary(positive_ratio)

    st.subheader("종합 평가")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("👍 긍정적", f"{positive_percent}%")
    with col2: st.metric("👎 부정적", f"{negative_percent}%")
    with col3:
        if color_code == "blue": st.info(f"🏆 {summary_text}")
        elif color_code == "green": st.success(f"😊 {summary_text}")
        elif color_code == "orange": st.warning(f"😐 {summary_text}")
        else: st.error(f"😡 {summary_text}")

    st.progress(positive_ratio, text=f"긍정 리뷰 비율: {positive_percent}%")

    st.divider()
    st.subheader(f"리뷰 목록 ({total_reviews}개)")

    for idx, row in game_reviews.iterrows():
        play_hours = round(row['playtime_forever'] / 60, 1)
        is_positive = row['corrected_voted_up']
        
        with st.container(border=True):
            c1, c2 = st.columns([1, 5])
            with c1:
                if is_positive: st.success("👍 추천")
                else: st.error("👎 비추천")
                st.caption(f"⏱️ {play_hours}시간")
            with c2:
                st.write(row['review'])

    st.divider()
    if st.button("⬅️ 목록으로 돌아가기 (Bottom)", key="bottom_back", use_container_width=True):
        st.session_state.selected_game_id = None
        st.rerun()

# --- 목록 페이지 ---
else:
    if search_query:
        # unique_games가 이제 정의되어 있으므로 오류가 나지 않습니다.
        display_games = [g for g in unique_games if search_query.lower() in g.lower()]
    else:
        display_games = unique_games

    st.subheader("등록된 게임 목록")
    
    if len(display_games) == 0:
        st.warning("검색 결과가 없습니다.")
    
    cols = st.columns(2)
    for idx, g_name in enumerate(display_games):
        col = cols[idx % 2]
        with col:
            with st.container(border=True):
                img_url = GAME_IMAGES.get(g_name, GAME_IMAGES["기본 이미지"])
                st.image(img_url, use_container_width=True)
                
                if img_url == GAME_IMAGES["기본 이미지"]:
                    st.caption(f"⚠️ 실제 이름: {g_name}")

                if st.button(f"🔎 {g_name}", key=f"btn_{g_name}", use_container_width=True):
                    show_game_details(g_name)
                    st.rerun()