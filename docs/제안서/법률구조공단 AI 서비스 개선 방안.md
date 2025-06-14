대한민국 법률구조공단 AI 서비스 개선을 위한 기술적 타당성 심층 분석: 오픈소스 기반 구현 가능성, 현황 분석, 사용자 중심 문제 해결 및 맞춤형 제안 연구I. Executive Summary본 보고서는 대한민국 법률구조공단(이하 공단)의 AI 서비스 개선을 위한 기술적 타당성을 심층적으로 분석하고, 오픈소스 기반의 구현 가능성을 탐색하며, 사용자 중심의 문제 해결 및 맞춤형 제안을 위한 AI 서비스 구현 방안을 연구하는 것을 목표로 한다. 현재 공단은 AI 챗봇 및 콜봇 서비스를 운영하며 법률 정보 접근성 향상을 도모하고 있으나, 보다 고도화되고 개인화된 대화형 법률 지원 서비스에 대한 요구가 증대되고 있다.주요 분석 결과, 공단은 "법률똑똑이"와 같은 AI 기반 Q&A 서비스를 통해 법률 정보 제공의 기초를 다졌으나, 이는 일반적인 정보 제공 수준에 머무르고 있어 개인 맞춤형 심층 지원에는 한계가 있음이 확인되었다. 그러나 국내외적으로 활용 가능한 오픈소스 한국어 거대언어모델(LLM), 풍부한 법률 데이터셋(특히 AI Hub 제공 자료), 그리고 LangChain, Rasa와 같은 AI 개발 프레임워크의 발전은 제안된 서비스 개선이 기술적으로 충분히 실현 가능함을 시사한다. 사용자 중심의 문제 해결을 위해 AI는 자가 해결 지원, 외부 전문가 연계, 소송 절차 안내 등 다양한 역할을 수행할 수 있으며, 특히 한국 법률 지식그래프 구축은 개인 맞춤형 법률 정보 서비스 제공의 핵심 동력이 될 것이다.본 보고서는 공단 AI 서비스 개선을 위한 단계적 이행 로드맵을 제안하며, 초기에는 기존 Q&A 시스템 고도화 및 사용자 피드백 시스템 구축에 집중하고, 중장기적으로는 법률 지식그래프 구축 및 이를 활용한 개인 맞춤형 대화형 AI 에이전트 개발로 나아갈 것을 권고한다. 이 과정에서 데이터 거버넌스 확립, 지속적인 모델 개선을 위한 피드백 루프 구축, 그리고 법률 정보와 법률 조언 간의 명확한 경계 설정 등 윤리적 지침 준수가 필수적이다. 본 연구는 공단이 AI 기술을 통해 법률구조 서비스의 질을 한 단계 높이고, 국민의 사법 접근성을 혁신적으로 개선하는 데 기여할 수 있는 구체적인 기술적 방향을 제시한다.II. 대한민국 법률구조공단(KLAC) AI 서비스 현황 분석A. 기존 공단 AI 이니셔티브 개요대한법률구조공단은 국민의 법률 서비스 접근성 강화를 목표로 AI 기술 도입을 점진적으로 추진해왔다. 이러한 노력의 일환으로 다양한 AI 기반 서비스가 운영 중이거나 계획 단계에 있다.1. "법률똑똑이" 챗봇공단의 대표적인 AI 서비스인 "법률똑똑이"는 사용자의 법률 관련 질의에 응답하는 인공지능 기반 Q&A 서비스이다.1 이 서비스는 가족관계등록, 주택/상가 임대차, 상속과 유언, 개인회생, 파산/면책, 친족 등 특정 법률 분야에 대한 정보를 제공하며, 방문 또는 화상 상담 예약 기능도 함께 지원한다.1 그러나 "법률똑똑이"는 명시적으로 "법률 질의에 대한 참고사례를 제공한 것으로 정확한 법률이 적용된 법률상담이 아니므로 개인 간의 분쟁 또는 법률절차에 적용할 수 없다"고 한계를 밝히고 있다.1 이는 현재 서비스가 일반적인 정보 제공에 초점을 맞추고 있으며, 개인의 구체적인 상황에 맞는 맞춤형 법률 조언과는 거리가 있음을 시사한다. 이러한 제약은 향후 AI 서비스 고도화를 통해 해결해야 할 주요 과제 중 하나로, 단순 정보 전달을 넘어선 심층적 안내 및 지원 기능의 필요성을 부각시킨다.2. 콜봇 및 일반 챗봇 서비스공단은 2023년 12월, 법률 서비스 이용 편의성을 높이기 위해 콜봇 및 챗봇 서비스를 확대 개시하였다.2 이러한 서비스는 대표번호 132를 통한 전화 상담 및 카카오톡 채널을 통해 접근 가능하다.3 이는 사용자와의 접점을 다양화하고 초기 문의에 대한 AI의 응대 범위를 넓히려는 시도로 해석된다. 다만, 현재 제공되는 정보에 따르면 이들 서비스가 "법률똑똑이"의 기능을 단순히 다른 채널로 확장한 것인지, 아니면 더 발전된 형태의 대화형 AI 기능을 포함하고 있는지에 대한 구체적인 기술적 설명은 부족하다. 서비스 개선 요구는 이들 챗봇/콜봇이 아직 기본적인 질의응답이나 단순 안내 수준에 머물러 있을 가능성을 시사하며, 사용자 질의의 복잡성과 다양성을 처리하고 개인화된 지원을 제공하는 데에는 한계가 있을 수 있다.3. AI 어드바이저 서비스콜봇, 챗봇과 함께 "AI 어드바이저" 서비스가 언급되었으나 2, 제공된 자료 내에서는 해당 서비스의 구체적인 기능이나 역할에 대한 상세 정보가 부족하다. 이 서비스의 실제 운영 현황과 기술적 특징에 대한 추가적인 조사가 필요하나, 명칭으로 미루어 볼 때 단순 정보 제공을 넘어선 일정 수준의 안내 또는 조언 기능을 목표로 할 가능성이 있다.4. 통합 플랫폼 비전공단은 2023년부터 2025년까지 전자문서 기반의 비대면 법률구조 서비스 접수, 누적된 법률상담 데이터를 활용한 AI 기반 법률정보 안내 및 유사 사례 소개, 그리고 다부처·다기관 연계를 통한 통합 플랫폼 서비스 구현을 추진 중이다.5 이 계획에 따르면, 2024년까지 AI를 통해 간단하고 신속한 법률 서비스를 제공하는 것을 목표로 하고 있다.5 이러한 장기적 비전은 본 보고서에서 제안하는 AI 서비스 개선 방향과 일치하며, 제안된 기술들은 공단의 통합 플랫폼 전략을 구체화하고 가속화하는 데 기여할 수 있다. 특히 "누적된 법률상담 데이터"의 활용 계획은 고품질의 한국어 법률 AI 모델 학습 및 RAG 시스템 구축에 매우 중요한 자산이 될 수 있다.B. 현재 사용자 상호작용 지점 및 서비스 제공 방식현재 공단 사용자는 다양한 채널을 통해 법률 정보 및 상담 서비스를 이용할 수 있다. AI는 주로 초기 정보 탐색 단계에서 활용되며, 복잡하거나 개인화된 지원이 필요한 경우 인간 상담원으로 연결되는 구조를 가지고 있다.
정보 접근 채널: 사용자는 공단 웹사이트의 "법률똑똑이" 챗봇, 카카오톡 채널의 챗봇, 그리고 대표번호 132를 통한 전화(콜봇 또는 인간 상담원 연결)를 통해 AI 서비스와 상호작용한다.1
상담 방식: AI를 통한 초기 정보 습득 외에, 사무실 내방을 통한 면접 상담, 웹사이트를 통한 사이버 상담 및 채팅 상담, 전화를 통한 상담 등 전통적인 방식의 인간 상담원 지원도 병행되고 있다.3
자가 해결 대 지원 서비스: 현재 AI 서비스는 주로 사용자가 스스로 정보를 찾아볼 수 있도록 지원하는 자가 해결 도구의 성격이 강하다. "법률똑똑이"의 경우, 제공되는 정보가 참고 사례 수준이며 법적 분쟁 해결에 직접 적용할 수 없음을 명시하고, 정확한 법률 상담은 방문 또는 전화 상담을 이용하도록 안내하고 있다.1 이는 AI의 역할이 아직은 정보 제공에 국한되어 있으며, 문제 진단이나 해결 경로 제시에 있어서는 제한적임을 나타낸다.
C. 사용자 질의 기반 파악된 격차 및 개선 영역사용자 질의에서 도출된 AI 서비스 개선 요구는 현재 공단 AI 서비스의 몇 가지 주요 격차를 시사한다.
제한된 개인화 수준: 현재 서비스는 일반적인 법률 정보를 제공하는 데 그치고 있어, 사용자의 구체적인 상황과 필요에 맞춘 개인화된 안내나 정보 제공이 미흡하다. "법률똑똑이"의 면책 조항은 이러한 한계를 명확히 보여준다.1
기본적인 Q&A 기능: 기존 Q&A 시스템이 존재하지만, 사용자 질의는 보다 정교하고 문맥을 이해하는 검색 증강 생성(RAG) 기반 시스템의 필요성을 암시한다. 이는 단순 키워드 매칭을 넘어선 심층적인 정보 검색 및 생성을 의미한다.
능동적 안내 부족: 현재 AI는 사용자가 특정 정보를 요청할 때 수동적으로 응답하는 경향이 있다. 사용자의 문제를 진단하고, 소액사건과 같은 복잡한 절차를 단계별로 안내하거나, 필요한 다음 단계를 능동적으로 제시하는 기능이 부족하다.
단편적인 지원 경로: 다양한 상담 채널이 존재하지만, AI가 사용자의 상황을 종합적으로 판단하여 가장 적합한 내부 상담 서비스(예: 특정 분야 전문 상담원)나 외부 유관기관으로 지능적으로 연결하는 기능이 강화될 필요가 있다.
데이터 활용 잠재력 미흡: 공단은 방대한 양의 법률 상담 데이터를 축적하고 있을 것으로 예상되며 5, 이는 고도로 특화된 한국어 법률 AI 모델을 학습시키고 RAG 시스템의 지식 기반을 구축하는 데 매우 귀중한 자원이다. 그러나 현재 이러한 데이터가 AI 서비스 고도화에 충분히 활용되고 있는지는 미지수이다. 적절한 익명화와 윤리적 고려를 전제로 이 데이터를 활용한다면, AI 서비스의 질을 획기적으로 향상시킬 수 있다.
결론적으로, 공단의 AI 서비스는 정보 제공의 초기 단계를 넘어, 사용자의 법률 문제 해결 여정 전반에 걸쳐 보다 적극적이고 지능적인 역할을 수행하는 방향으로 개선될 필요가 있다. 이는 AI를 단순 정보 검색 도구에서 개인 맞춤형 법률 길잡이로 전환하는 것을 의미한다.III. 오픈소스 기반 AI 서비스 개선의 기술적 타당성공단 AI 서비스의 사용자 중심적 개선은 오픈소스 기술을 적극적으로 활용함으로써 기술적 타당성을 확보할 수 있다. 한국어 자연어 처리(NLP) 기술의 발전, 풍부한 법률 데이터셋의 가용성, 그리고 고도화된 AI 개발 프레임워크의 등장은 이러한 개선을 뒷받침하는 핵심 요소이다.A. 한국어 법률 AI를 위한 오픈소스 기반 기술1. 한국어 거대언어모델(LLM): 오픈소스 옵션 평가최근 한국어 NLP 분야는 상당한 발전을 이루었으며, 법률 분야에 적용 가능한 다양한 오픈소스 LLM이 등장하고 있다. 모델 선택 시에는 성능, 파라미터 크기(미세조정 가능성 및 운영 비용 고려), 라이선스 등을 종합적으로 고려해야 한다.6 특히 70억 파라미터(7B) 미만의 sLLM(Small Language Model)은 미세조정의 현실성과 효율성 측면에서 우선적으로 검토될 수 있으며, 더 큰 모델의 경우에도 법률 분야에서 뛰어난 제로샷/퓨샷 성능을 보인다면 고려 대상이 될 수 있다.

주요 고려 모델:

KoSimCSE 계열 (daekeun-ml/KoSimCSE-supervised-roberta-base, BM-K/KoSimCSE-roberta 등): 문장 임베딩 모델로서, RAG 시스템의 의미론적 검색 성능에 핵심적인 역할을 한다. 문장 간 유사도 이해에 강점을 보인다.7 특히 BM-K/KoSimCSE-roberta는 1억 1100만(111M) 파라미터로 한국어 의미 유사도 평가에서 높은 성능을 기록했으며 8, daekeun-ml/KoSimCSE-supervised-roberta-base는 지도 학습 기능을 제공하여 특정 자연어 추론(NLI) 데이터셋 기반 미세조정에 유용하다.10 이러한 모델들은 RAG 파이프라인의 검색(retrieval) 구성요소 구축에 필수적이며, 상대적으로 작은 크기와 의미 이해 능력은 법률 문서 검색에 적합하다.
Polyglot-Ko 계열 (EleutherAI/polyglot-ko-1.3b, EleutherAI/polyglot-ko-5.8b 등): Q&A 및 특허 데이터를 포함한 방대하고 다양한 한국어 데이터셋으로 학습된 자동회귀 모델이다.12 13억(1.3B) 파라미터 모델 12은 크기와 학습 데이터의 질을 고려할 때 미세조정 후보로 적합하며, 58억(5.8B) 모델 14은 더 큰 용량을 제공하지만 더 많은 자원을 필요로 한다. 이 모델들은 다양한 한국어 텍스트, 특히 Q&A와 특허 문서와 같은 정형화된 언어에 대한 이해도가 높아 법률 분야 적용에 유리하며, Apache 2.0 라이선스는 활용에 용이하다.
Korean LLM Project (quantumaikr/KoreanLM): 다수 모델의 영어 중심성을 극복하고 한국어 토큰화 효율성을 개선하기 위해 다양한 크기의 한국어 최적화 LLM 제공을 목표로 한다.17 한국어 최적화와 다양한 모델 크기 제공은 공단의 특정 요구사항에 맞춰 모델을 선택하고 활용하는 데 이점을 제공한다.
오픈소스 소형 언어 모델 (sLLM) (예: KoAlpaca, KULLM, SOLAR mini 등):

KoAlpaca (예: beomi/KoAlpaca-Polyglot-12.8B (7B 초과 모델이나, 다양한 변형 존재 가능), LLaMA 기반): 한국어 명령 이해를 위해 설계된 지시사항 조정(instruction-tuned) 모델이다.18 LLaMA 기반의 더 작은 버전들이 존재할 가능성이 있다.
KULLM (nlpai-lab/kullm-polyglot-12.8b-v2 (7B 초과 모델이나, 다양한 변형 존재 가능)): 고려대학교에서 개발한 한국어 특화 대화 모델이다.18
SOLAR mini (upstage/SOLAR-mini-250422 - 10.7B 모델이나 "경량"으로 언급되며 한국어 성능 우수, 다른 "mini" 버전은 더 작거나 효율적일 수 있음): 업스테이지에서 개발했으며, 한국어 LLM 리더보드에서 높은 성능을 보인다.18 107억(10.7B) 파라미터 모델이지만, 효율성 측면에서 더 작은 모델과 자원 요구량이 비슷할 수 있다.
GECKO-7B (kifai/GECKO-7B): 한국어, 영어, 코드로 사전 학습된 70억(7B) 파라미터 모델이다.22
EEVE-Korean-Instruct-7B-v2.0-Preview (yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview): Qwen2.5-7B 기반으로 한국어 이해 및 생성 능력이 강화되었다.23
OLAIR/ko-r1-7b-v2.0.3: 76.1억(7.61B) 파라미터, Qwen2 기반으로 추론 능력에 중점을 둔다.24
지시사항 조정 모델들은 일반적으로 Q&A 및 작업 지향 챗봇 개발에 더 나은 시작점을 제공한다. 7B 파라미터 내외의 고성능 모델 중 Apache 2.0 또는 유사한 허용적 라이선스를 가진 모델을 찾는 것이 중요하다.





미세조정(Fine-tuning) 고려사항:

법률 분야 특수성: 일반적인 한국어 LLM은 특정 법률 용어, 뉘앙스, 추론 패턴을 이해하기 위해 법률 말뭉치 기반의 미세조정이 필수적이다.25 KorFinMTEB 연구 26는 금융 텍스트 분야에서도 도메인 특화 벤치마크와 모델의 필요성을 강조하며, 이는 법률 텍스트에도 유사하게 적용될 수 있다. 단순히 벤치마크를 번역하거나 일반 모델을 사용하는 것만으로는 전문 분야에서 충분한 성능을 기대하기 어렵다.
지시사항 조정(Instruction Tuning): Q&A 및 챗봇 기능 구현을 위해서는 지시사항 조정 데이터셋(예: 법률 Q&A 쌍)을 활용한 미세조정이 매우 중요하다.
자원 요구사항: sLLM을 미세조정하는 데에도 상당한 계산 자원(GPU, 메모리)과 전문 지식이 필요하다.
HuggingFace 생태계: 미세조정을 위한 다양한 도구와 플랫폼을 제공한다.29


결론적으로, 검색에는 KoSimCSE와 같은 강력한 한국어 문장 임베딩 모델을, 답변 생성 및 대화 기능에는 법률 도메인에 맞게 미세조정된 한국어 sLLM(가능하다면 7B 미만, 또는 자원이 허용된다면 SOLAR mini와 같이 효율적인 대형 모델)을 사용하는 하이브리드 접근 방식이 최적일 수 있다. 핵심 과제는 일반 모델을 고도로 전문화된 법률 도메인에 효과적으로 적용하는 것이다.2. 한국어 법률 말뭉치 및 데이터셋: 오픈소스 자원 평가효과적인 법률 AI 서비스 구축을 위해서는 양질의 한국어 법률 데이터셋 확보가 필수적이다. 다행히 AI Hub를 중심으로 다양한 법률 관련 텍스트 데이터가 공개되어 있어, 이를 RAG 시스템의 지식 기반 구축 및 LLM 미세조정에 적극 활용할 수 있다.

AI Hub 데이터셋:

법률 지식베이스 (dataSetSn=29): 법령, 조문, 판례, 법률 상담 데이터 등 약 27만 건의 데이터를 포함하며, 특히 "생활법령" 데이터 내에는 Q&A 형식의 자료가 포함되어 있다.34 이는 RAG 시스템의 지식 기반 구축 및 Q&A 모델 미세조정에 직접적으로 활용될 수 있는 매우 가치 있는 자원이다.
법률/규정 (판결서, 약관 등) 텍스트 분석 데이터 (dataSetSn=580): 1만 건 이상의 판결문(민사, 형사, 행정)과 1만 건 이상의 약관 데이터로 구성되며, 사실관계, 주장, 결정, 약관 조항의 유불리 판단 등 상세한 태깅 정보가 포함되어 있다.36 이 데이터는 RAG 시스템 구축, 법률 텍스트 이해 모델 미세조정뿐만 아니라, 요약이나 논증 추출과 같은 특정 작업에도 유용하게 사용될 수 있다. 특히 구조화된 정보(청구, 사실, 추론 등)는 모델 학습에 큰 이점을 제공한다.
문서요약 텍스트 데이터 (dataSetSn=123 또는 AI Hub ID 8054): 법원 판결문 3만 건을 포함하여 추출 요약(3문장) 및 생성 요약문을 제공한다.38 또 다른 데이터셋(dataSetSn=452 또는 AI Hub ID 452)은 법률 문서에서 핵심 3문장을 추출하는 데 중점을 둔다.39 이러한 데이터셋은 법률 문서 요약 모델 학습 및 미세조정에 필수적이다.
생성형AI 법률/규정 텍스트 분석 데이터 (고도화) (dataSetSn=71723): 약 25만 건의 상황별 판례 데이터와 추출 요약, 그리고 7만 784건의 Q&A 세트, 추가로 전문가가 직접 생성한 2만 160건의 Q&A 세트를 포함한다.40 이 데이터셋은 방대한 규모와 판례 기반 Q&A에 초점을 맞추고 있어 RAG 시스템 및 고도화된 Q&A 모델 미세조정에 매우 유용하다.
생성형AI 법률 지식기반 관계 데이터 (dataSetSn=71722): 세법 분야 중심의 법률 문서 3만 9,035건에 대해 트리플 라벨링(주체-관계-객체)을 수행하여 지식그래프 구축을 위한 데이터를 제공한다.35 이는 특히 세법 분야의 한국 법률 지식그래프 구축에 직접 적용 가능하며, 다른 법률 분야로 확장할 수 있는 방법론적 기반을 제공한다.
AI Hub 데이터 일반 참고사항: AI Hub 데이터셋은 일반적으로 비상업적 연구 목적으로 제공되나, 공공기관인 공단은 더 넓은 활용 권한을 가지거나 협상을 통해 이를 확보할 수 있을 것이다. 데이터는 주로 한국어로 제공된다.34



KoNLPy 말뭉치 (kolaw, kobill):

kolaw: 대한민국 헌법 텍스트를 포함한다.42
kobill: 국회 의안 텍스트를 포함한다.42
이들 말뭉치는 AI Hub 데이터에 비해 규모는 작지만, 일반적인 법률 언어 이해 모델 학습 데이터 보강에 활용될 수 있다. GPL 라이선스를 따른다.



KRLawGPT의 CKLC (Clean Korean Legal Corpus):

KRLawGPT 사전 학습에 사용된 "포괄적인 한국 법률 데이터셋"으로 설명되나 44, 구체적인 내용, 규모, 외부 사용 가능 여부는 제공된 자료에서 확인되지 않아 추가 조사가 필요하다. 접근 가능하다면 매우 유용한 자원이 될 수 있다.


이처럼 AI Hub를 중심으로 풍부한 한국어 법률 텍스트 데이터가 존재하며, 이는 공단 AI 서비스 개선을 위한 RAG 지식 기반 구축 및 LLM 미세조정에 충분한 기반을 제공한다. 특히 "고도화"된 Q&A 데이터셋 40은 서비스의 질을 크게 향상시킬 잠재력을 가지고 있다. 이러한 다양한 데이터셋을 효과적으로 처리하고 통합하여 일관된 지식 기반을 구축하고, 각 AI 기능의 특성에 맞는 적절한 데이터를 선별하여 미세조정에 활용하는 전략이 중요하다.3. 법률 정보 검색을 위한 검색 증강 생성(RAG) 프레임워크 및 기술법률 분야의 정확하고 신뢰할 수 있는 정보 제공을 위해서는 RAG(Retrieval Augmented Generation) 파이프라인 구축이 핵심적이다. RAG는 대규모 외부 지식베이스에서 관련 정보를 검색하여 LLM의 답변 생성을 보강하는 방식으로, LLM의 환각(hallucination) 현상을 줄이고 최신 정보 및 도메인 특화 지식을 반영하는 데 효과적이다.

LangChain 프레임워크: RAG 파이프라인 구축을 위한 대표적인 오픈소스 프레임워크이다.45 문서 로더(PDF, 웹 등), 텍스트 분할기(문자, 재귀, 의미 기반), 임베딩 모델 통합, 벡터 저장소, 검색기, Q&A 체인 등 RAG 구현에 필요한 핵심 구성요소를 제공한다.45 LangChain 자체는 언어에 구애받지 않으며, 한국어 처리 성능은 선택된 LLM, 임베딩 모델, 토크나이저에 따라 달라진다. 최근 한국어 관련 튜토리얼과 자료도 증가하고 있다.46


벡터 데이터베이스: 검색된 법률 문서의 임베딩을 저장하고 효율적인 유사도 검색을 수행하기 위해 오픈소스 벡터 데이터베이스(예: ChromaDB, FAISS)를 LangChain과 함께 활용할 수 있다.45


검색 기술:

밀집 검색(Dense Retrieval): KoSimCSE와 같은 임베딩 모델을 사용하여 의미론적 유사도에 기반한 검색을 수행한다. 이는 RAG의 표준적인 검색 방식이다.
희소 검색(Sparse Retrieval, 예: BM25): TF-IDF와 유사한 키워드 기반 검색 방식으로, 계산 비용이 낮고 특정 법률 용어 검색에 효과적일 수 있다.56 다만, 한국어 금융 도메인에서의 BM25 적용 결과, 때때로 극단적인 점수를 반환하여 임베딩 기반 방식이 미묘한 의미 검색에 더 안정적일 수 있음을 시사했다.59
하이브리드 검색 (Reciprocal Rank Fusion - RRF): 밀집 검색과 희소 검색 결과를 결합하여 각 방식의 장점을 활용하는 기법이다. OpenSearch는 RRF를 지원하며 60, 이는 의미론적 이해와 법률 용어의 키워드 정밀도를 결합할 수 있어 법률 검색에 매우 유용할 수 있다.
지연 상호작용 모델 (Late Interaction Models, 예: ColBERTv2): jina-colbert-v2는 89개 언어(한국어 포함 가능성 높음)를 지원하는 다국어 모델로, 압축된 임베딩과 강력한 검색 성능을 제공한다.62 강력한 성능을 제공하지만, 특수한 인덱싱 과정이 필요할 수 있으며, 비상업적 버전이 HuggingFace를 통해 제공된다.



재순위화(Re-ranking):

초기 검색기(bi-encoder)가 검색한 문서들의 관련성을 향상시키기 위해 크로스 인코더(cross-encoder)를 사용하여 재순위화를 수행할 수 있다.64
한국어 특화 재순위화 모델인 ko-reranker-8k (upskyy/ko-reranker-8k)는 BAAI/bge-reranker-v2-m3 모델을 한국어 데이터로 미세조정한 것이다.65



분할(Chunking) 전략: 길고 복잡한 법률 문서 처리에 매우 중요하다. 고정 크기 분할, 재귀적 분할, 의미론적 분할 등 다양한 전략을 고려할 수 있으며 45, 법률 문서의 경우 조항, 항, 목 등 의미 단위 기반의 문서 구조적 분할이 이상적일 수 있다.


인용 및 출처 추적: 법률 AI에서는 답변의 투명성과 검증 가능성을 확보하기 위해 출처 인용이 필수적이다. LangChain과 LlamaIndex는 이러한 기능을 구현하기 위한 메커니즘을 제공한다.66

법률 분야는 높은 정확성과 근거 제시가 요구되므로, 단순한 RAG 시스템을 넘어선 고도화된 파이프라인 구축이 필요하다. 예를 들어, 1) 법률 조항이나 판례의 의미 단위를 고려한 지능적 분할, 2) 특정 법률 용어 검색을 위한 BM25와 의미론적 유사도 검색을 결합한 하이브리드 검색(RRF 활용), 3) ko-reranker-8k와 같은 한국어 특화 재순위화 모델 적용, 4) 법률 질의응답에 미세조정된 한국어 LLM을 통한 답변 생성, 그리고 5) LlamaIndex 등에서 제공하는 강력한 출처 인용 기능 68 통합 등을 고려할 수 있다. 이러한 구성 요소들을 한국 법률 언어의 특성에 맞게 효과적으로 통합하고 최적화하는 것이 핵심 과제가 될 것이다.B. AI 기반 사용자 중심 문제 해결: 구현 전략공단 AI 서비스 개선의 핵심은 사용자가 직면한 법률 문제를 효과적으로 해결할 수 있도록 지원하는 사용자 중심적 접근 방식을 채택하는 것이다. 이는 AI가 단순 정보 제공자를 넘어, 문제 진단, 해결 경로 탐색, 필요시 전문가 연계까지 안내하는 적극적인 역할을 수행하도록 설계하는 것을 의미한다.1. 자가 해결 능력 강화

고도화된 법률 Q&A 시스템:

구현: 앞서 논의된 RAG 파이프라인을 기반으로, AI Hub의 방대한 법률 데이터셋 34 및 기타 법률 문서를 활용하여 지식 베이스를 구축한다. 한국어 sLLM을 법률 질의 이해 및 정보성 답변 생성에 맞게 미세조정한다.
프롬프트 엔지니어링: 양질의 법률 질문과 답변 예시를 제공하는 퓨샷 프롬프팅(Few-Shot Prompting) 70, 복잡한 법률 추론 과정을 단계별로 안내하는 연쇄적 사고 프롬프팅(Chain-of-Thought Prompting, CoT) 74 (단, 정보 제공에 국한되도록 주의), 그리고 "당신은 공단의 유용한 법률 정보 안내 AI입니다"와 같이 AI의 역할을 명확히 정의하는 역할 프롬프팅(Role Prompting) 76 등 고급 프롬프트 엔지니어링 기법을 적용한다.
Q&A 데이터셋 구축 방법론: AI Hub에서 제공하는 기존 Q&A 데이터셋 34을 적극 활용한다. 새로운 Q&A 쌍 구축 시에는 공단에 자주 접수되는 사용자 문의 유형과 공신력 있는 법률 문서를 분석하여 질문-답변 쌍을 생성하며, 초기 생성에는 LLM을 활용하되 반드시 법률 전문가의 검토를 거친다.25 사용자가 자신의 문제를 정확한 법률 용어로 표현하지 못하는 경우가 많으므로, AI는 대화형 Q&A와 지식그래프(후술) 참조를 통해 사용자의 실제 법률적 쟁점을 추론해내는 능력을 갖추어야 한다.



자동 법률 문서 요약:

구현: AI Hub의 법률 문서 요약 데이터셋 38 (특히 dataSetSn=123 또는 8054는 3만 건의 판결문 요약 포함)을 활용하여 한국어 sLLM(예: Polyglot-Ko, SOLAR mini)을 미세조정한다.
활용 사례: 사용자가 이해하기 어려운 긴 판례, 법령 조항, 복잡한 법률 해설 등을 간결하게 요약하여 제공함으로써 정보 접근성을 높인다.



초기 법률 문제 진단 (사용자 여정 및 문제 평가):

구현: Rasa 또는 LangChain Agents와 같은 대화형 AI 프레임워크를 사용하여 사용자와의 단계별 질의응답을 통해 상황을 파악하는 대화 흐름을 설계한다. 이는 의료 분야의 "가상 분류(virtual triage)" 85와 유사한 접근 방식이다.
사용자 여정 매핑: AI는 사용자의 문제를 알려진 법률 범주와 연결하는 데 도움을 주어야 한다.86 이는 사용자의 필요, 고충 지점, 정보 탐색 방식을 이해하는 것을 포함한다.
결과: 진단 결과를 바탕으로 AI는 관련된 법률 주제, 자가 해결 자료, 또는 전문가 상담 필요성을 제안할 수 있다.90 예를 들어, 사용자가 "집주인이 보증금을 안 돌려줘요"라고 입력하면, AI는 이를 '임대차 보증금 반환 분쟁'으로 잠정 진단하고 관련 법 조항, 해결 절차(내용증명, 지급명령, 소액소송 등)에 대한 정보를 제공하거나, 사안의 복잡성에 따라 공단 상담 예약을 안내할 수 있다.


이러한 자가 해결 기능 강화는 사용자가 법률 문제에 직면했을 때 초기 정보 탐색의 부담을 줄이고, 자신의 상황을 객관적으로 이해하며, 가능한 해결 경로를 모색하는 데 실질적인 도움을 줄 수 있다.2. 외부 및 인적 지원과의 지능적 연계AI는 사용자가 자가 해결하기 어려운 문제에 직면했을 때, 적절한 외부 자원이나 공단 내 인적 지원으로 원활하게 연결하는 지능형 게이트웨이 역할을 수행해야 한다.

챗봇 기반 안내 및 추천 시스템:

구현: Rasa 93 또는 LangChain Agents 48와 같은 대화형 AI 프레임워크를 활용한다.
로직: 초기 법률 문제 진단 결과를 바탕으로 AI는 자가 해결 가능성을 판단한다. 자가 해결이 어렵다고 판단될 경우, 다음과 같은 조치를 취할 수 있다:

공단의 특정 인적 상담 서비스(전화, 방문, 화상)로 안내하고, 경우에 따라 상담 예약 절차를 지원한다.1
문제가 공단의 직접적인 지원 범위를 벗어나거나 공단 자원이 부족할 경우, 관련된 다른 공공기관 또는 비영리 법률구조 단체 정보를 제공하고 연결을 안내한다.
원활한 상담 이관을 위해 공단의 기존 CRM 또는 사건 관리 시스템과의 연동을 고려한다.118





실행 가능한 체크리스트 자동 생성:

구현: 진단된 법률 문제 유형(예: 소액사건심판 준비, 법률구조 신청)에 따라, AI가 필요한 서류 목록, 수집해야 할 정보, 절차적 단계 등을 포함하는 맞춤형 체크리스트를 생성하여 제공한다.119
라이브러리 활용: 체크리스트 UI 생성에는 dash.dcc.Checklist 123와 같은 Python 라이브러리를 활용할 수 있으나, 체크리스트 내용 생성 로직은 구조화된 지식이나 미세조정된 LLM으로부터 도출되어야 한다. 예를 들어, 소액사건 제소를 원하는 사용자에게는 AI가 "1. 소장 작성 (필요 정보: 원고/피고 인적사항, 청구취지, 청구원인), 2. 증거자료 준비 (계약서, 영수증 등), 3. 관할 법원 확인, 4. 인지대 및 송달료 납부"와 같은 구체적인 체크리스트를 제공할 수 있다.


이러한 지능적 연계 기능은 사용자가 자신의 문제에 맞는 최적의 지원을 신속하게 받을 수 있도록 돕고, 공단 내부적으로는 상담 자원의 효율적 배분에 기여할 수 있다. AI는 단순 정보 전달자를 넘어, 문제 해결 여정의 능동적인 네비게이터 역할을 수행하게 된다.3. AI 기반 소송 지원 (사용자 역량 강화 중심)비교적 정형화되어 있고 절차가 명확한 특정 소송 유형에 대해 AI는 사용자가 스스로 절차를 이해하고 준비하는 데 필요한 정보를 제공하여 역량을 강화할 수 있다. 이는 특히 변호사 선임 없이 직접 소송을 진행하려는 사용자에게 유용하다.

소액사건심판 절차 안내:

현재 기준금액: 2025년 초 현재 소액사건심판의 대상이 되는 소송목적의 값(청구금액 상한)은 3,000만원이다.124 (과거 2,000만원 기준 126은 현재 적용되지 않음).
정보 제공: AI는 다음과 같은 소액사건심판 절차를 단계별로 설명할 수 있다:

원고의 소장 접수: 법원에 소장을 제출한다.127
법원의 이행권고결정: 법원은 소장을 검토 후 피고에게 채무 이행을 권고하는 결정을 내릴 수 있다.127
피고의 이의신청: 피고는 이행권고결정 등본을 송달받은 날부터 2주일(14일) 이내에 이의신청을 할 수 있다.127
이의신청 없을 시 확정: 피고가 기간 내 이의신청을 하지 않거나, 이의신청이 각하/취하되면 이행권고결정은 확정판결과 동일한 효력을 가진다 (강제집행 가능).127
이의신청 시 재판 진행: 피고가 이의신청을 하면 정식 소액사건재판 절차(피고에게 소장 부본 송달 및 변론기일 통지, 변론기일 진행, 판결 선고)가 진행된다.127


필요 서류 안내: 소장, 채권자 및 채무자의 주소 명확화 서류, 채권 주장 증빙서류, 인지대 및 송달료 납부 영수증 등의 필요성을 안내한다.128 대리인이 있는 경우 소송위임장 및 관계 증명 서류도 필요하다.
소액사건 대리인: 소액사건의 경우 당사자의 배우자, 직계혈족(부모, 조부모, 자녀, 손자 등), 형제자매는 법원의 허가 없이 소송대리인이 될 수 있음을 안내한다.125



공단 법률구조 신청 절차 지원:

정보 제공: 공단의 주요 서비스(무료 법률상담, 소송서류 무료 작성, 민·가사 사건 소송대리, 형사사건 무료 변호 등)에 대해 안내한다.129
신청 방법: 공단에 직접 신청하거나 농협 등 중계기관을 통해 대리 신청이 가능함을 알린다.129
필요 서류 안내: 법률구조신청서(공단 양식), 신분증, 주민등록표등본(세대주 및 세대원 포함), 농업인 증명서류(해당 시), 건강보험자격(득실)확인서 또는 건강보험증, 중위소득 확인서류(건강보험료 납부확인서, 소득금액증명원 등) 등을 안내한다.129 관련 서식은 공단 홈페이지에서 다운로드 가능하다.
처리 절차: 신청서 접수 → 공단의 사실조사(서류 보완 포함) → 법률구조 대상자 여부, 구조 타당성, 승소 및 집행 가능성 등 검토 → 구조 결정 또는 기각 결정 (기각 시 이의신청 가능) → 구조 결정 시 공단 소속 변호사 등에 의한 소송 수행 (소송비용 공단 우선 부담 후 상환 조건 있을 수 있음) 순으로 진행됨을 설명한다.131


이러한 AI 기반 안내는 사용자가 복잡하게 느낄 수 있는 법적 절차를 보다 쉽게 이해하고 필요한 준비를 하는 데 도움을 주어, 법률 서비스에 대한 접근 장벽을 낮추는 데 기여할 수 있다. 이는 공단 AI 서비스의 사용자 중심성을 강화하는 중요한 요소이다.C. 개인 맞춤형 법률 정보 서비스 구현사용자에게 진정으로 유용한 AI 법률 서비스를 제공하기 위해서는 일반적인 정보 제공을 넘어 개인의 상황과 필요에 맞는 맞춤형 정보를 제공하는 것이 중요하다. 이를 위해 한국 법률 지식그래프 구축과 고도화된 대화형 AI 기술의 접목이 핵심 전략이 될 수 있다.1. 한국 법률 지식그래프(KG) 구축 및 활용

타당성 및 필요성: 포괄적인 한국 법률 지식그래프 구축은 개인 맞춤형 정보 검색 및 제공에 있어 매우 중요한 전략적 자산이다. 지식그래프는 법령, 판례, 법률 용어, 사건 유형, 절차 등 다양한 법률 지식 요소들을 개체(entity)로 정의하고 이들 간의 관계(relationship)를 구조화하여 표현한다. 이를 통해 단순 키워드 검색이나 기본적인 의미 검색을 넘어선, 법률 지식 간의 복잡한 연관 관계를 탐색하고 추론하는 것이 가능해진다. 예를 들어, 특정 법률 조항이 어떤 판례와 연관되어 있고, 그 판례가 사용자의 특정 계약 분쟁 유형에 어떤 영향을 미칠 수 있는지 등을 파악하는 데 활용될 수 있다.


오픈소스 기반 구축 도구:

한국어 법률 텍스트 대상 개체명 인식(NER) 및 관계 추출(RE): 지식그래프 구축의 첫 단계는 법률 문서에서 주요 개체(예: 법령명, 판례번호, 당사자, 법률행위, 손해배상액 등)를 식별하고 이들 간의 관계(예: 'A법률 제X조는 Y판례를 인용한다', '원고는 피고에게 Z를 청구한다')를 추출하는 것이다. 이를 위해 특화된 모델이 필요하다. 일반적인 한국어 NER/RE 모델 132도 존재하지만, AI Hub의 판결문 36이나 법령 34 데이터와 같은 법률 텍스트로 미세조정하는 과정이 필수적이다. 최근 등장한 KGGen 141은 LLM을 사용하여 텍스트로부터 지식그래프를 추출하는 오픈소스 Python 도구로, 한국어 법률 문서에 적용을 검토해볼 수 있다.
지식그래프 데이터베이스: 추출된 개체와 관계는 오픈소스 그래프 데이터베이스(예: Neo4j 143)에 저장하여 효율적으로 관리하고 질의할 수 있다.
메타데이터 관리 플랫폼: DataHub 144나 Apache Atlas 152와 같은 오픈소스 메타데이터 관리 플랫폼은 지식그래프 내 법률 개체 및 관계의 메타데이터를 체계적으로 관리하는 데 도움이 된다. DataHub는 사용자 정의 개체(예: '법령', '판례') 및 속성(aspect)(예: '관할법원', '핵심키워드', '관련조항') 정의를 지원하며 147, Apache Atlas는 법률 문서 및 해당 메타데이터(예: '법률명', '판례번호')에 대한 사용자 정의 유형(type) 및 분류(classification) 체계 생성을 지원한다.155
KGTK (Knowledge Graph Toolkit): 대규모 지식그래프 생성 및 활용을 위한 오픈소스 프레임워크로, 다양한 변환 및 분석 기능을 제공한다.159



AI Hub 데이터 활용: AI Hub의 "생성형AI 법률 지식기반 관계 데이터" 41는 세법 분야에 특화되어 있지만, 트리플 라벨링(주체-관계-객체) 형태로 구축되어 있어 한국 법률 지식그래프 구축의 방법론적 참조 모델이 될 수 있다.


활용 방안: 구축된 법률 지식그래프는 RAG 시스템의 검색 정확도와 답변의 맥락적 적합성을 크게 향상시킬 수 있다. 사용자의 질의에 대해 관련된 법 조항, 판례, 해석, 유사 상담 사례 등을 지식그래프를 통해 유기적으로 연결하여 제공함으로써, 보다 심층적이고 개인화된 정보 제공이 가능해진다.

법률 지식그래프 구축은 상당한 초기 투자가 필요한 작업이지만, 장기적으로 공단 AI 서비스의 지능화 수준을 획기적으로 높일 수 있는 핵심 기반 기술이다. AI Hub의 관련 데이터셋 41과 오픈소스 도구들을 활용하여 특정 법률 분야부터 시범적으로 구축하고 점진적으로 확장하는 전략을 고려할 수 있다.2. 고도화된 대화형 AI: 문맥, 기억, 프롬프트 엔지니어링개인 맞춤형 법률 정보 서비스를 효과적으로 제공하기 위해서는 단순한 Q&A를 넘어 사용자와의 자연스러운 다회성 대화(multi-turn conversation)를 이해하고, 이전 대화 내용을 기억하며, 정교한 프롬프트 엔지니어링을 통해 LLM의 응답을 제어하는 고도화된 대화형 AI 기술이 필요하다.

챗봇 프레임워크:

Rasa: 오픈소스 대화형 AI 프레임워크로, 한국어 자연어 이해(NLU) (의도 인식, 개체 추출) 기능을 spaCy와 같은 구성요소나 사용자 정의 파이프라인을 통해 지원한다.93 Rasa의 사용자 정의 액션(custom actions)은 외부 API(예: RAG 파이프라인, 지식그래프 질의, 알림 서비스) 호출 기능을 구현하는 데 사용될 수 있다.99
LangChain: LLM과의 통합, 대화 메모리 관리 등 대화형 에이전트 구축을 위한 다양한 도구를 제공한다.42



대화 메모리(Conversation Memory): 다회성 법률 상담에서 문맥을 유지하기 위해 필수적이다.

LangChain은 다양한 메모리 유형(Buffer, Summary, Window, Knowledge Graph, Entity Memory 등)을 제공하며 116, RunnableWithMessageHistory는 대화 기록 관리를 위한 핵심 구성요소이다.50



법률 분야를 위한 정교한 프롬프트 엔지니어링:

퓨샷 프롬프팅(Few-Shot Prompting): 법률 질문과 바람직한 답변 스타일의 예시를 LLM에 제공하여 응답의 질을 높인다.70
연쇄적 사고 프롬프팅(Chain-of-Thought Prompting, CoT): 복잡한 법률 질의에 대해 LLM이 단계별로 논리적인 사고 과정을 거치도록 유도한다.74 (단, 법률 조언이 아닌 정보 제공 범위 내에서 활용)
역할 프롬프팅(Role Prompting): LLM에게 "당신은 대한민국 법률구조공단의 법률 정보 안내 AI입니다. 사용자의 질문에 대해 정확하고 중립적인 정보를 제공해야 합니다."와 같이 구체적인 역할, 페르소나, 범위, 한계 등을 명확히 지시한다.76
법률 특화 지시사항: 프롬프트는 달성하고자 하는 목표를 명확히 정의하고, 필요한 경우 정확한 법률 용어를 사용하며(LLM이 이를 이해하고 사용하도록 유도), 충분한 맥락(관련 사실관계 등)을 제공하고, 원하는 답변 형식을 지정해야 한다.76


진정으로 개인화된 서비스를 제공하기 위해서는 AI가 단순히 질문에 답하는 것을 넘어, 대화의 흐름을 이해하고, 세션 내 이전 상호작용을 기억하며, 법률 담론의 복잡성을 다룰 수 있도록 정교한 프롬프트를 통해 안내받아야 한다. 예를 들어, 사용자가 "부당해고"에 대한 일반적인 질문으로 시작한 후, 대화가 진행됨에 따라 AI는 (메모리 기능을 통해) 사용자가 제공한 세부 정보(예: 계약 유형, 고용 기간)를 기억한다. CoT 프롬프팅은 AI가 이러한 사실들을 관련 노동법 조항과 비교 분석하는 데 도움을 줄 수 있으며, 퓨샷 예제는 AI의 답변이 공단의 지침에 맞는 중립적이고 정보적인 어조를 유지하도록 보장할 수 있다.IV. 공단 AI 서비스 개선을 위한 전략적 제언공단 AI 서비스의 성공적인 개선을 위해서는 기술적 구현뿐만 아니라, 단계적 접근, 데이터 거버넌스 확립, 윤리적 사용 원칙 준수, 그리고 지속적인 개선을 위한 피드백 시스템 구축이 종합적으로 고려되어야 한다.A. 단계별 오픈소스 구현 로드맵AI 서비스 개선은 점진적이고 체계적인 접근을 통해 위험을 관리하고 각 단계에서 가시적인 성과를 도출하는 방식으로 추진하는 것이 바람직하다.

1단계: 기반 강화 (6-12개월)

기존 챗봇("법률똑똑이") 고도화: 사전 학습된 한국어 문장 임베더(예: KoSimCSE)와 범용 한국어 sLLM(예: Polyglot-Ko 1.3B 또는 적합한 지시사항 조정 모델 <7B)을 활용하여 RAG 파이프라인을 견고하게 구축한다. 기존 서비스 분야(가족관계, 임대차 등 1)에 대한 Q&A 정확도 향상에 집중한다.
법률 문서 요약 모듈 개발 및 통합: AI Hub의 법률 문서 요약 데이터셋 38을 활용하여 내부 업무용 또는 선별된 대국민 요약 정보 제공 기능을 개발한다.
기본 사용자 피드백 메커니즘 도입: Q&A 답변에 대한 간단한 피드백(예: 도움돼요/안돼요) 수집 기능을 Langfuse 168 또는 Chatwoot 26과 같은 오픈소스 도구를 활용하여 구현한다.



2단계: 사용자 중심 안내 및 지원 (12-24개월)

소액사건심판 AI 안내 절차 개발: 소액사건 관련 정보 125를 기반으로 AI 기반 단계별 안내 기능을 구현한다.
공단 법률구조 신청 AI 지원 기능 개발: 법률구조 신청 절차 및 필요 서류 129에 대한 AI 안내 기능을 제공한다.
선별된 한국어 sLLM 미세조정 시작: 공단 내부 Q&A 데이터(익명화 처리 후) 및 AI Hub 법률 Q&A 데이터셋 34을 활용하여 특정 법률 분야에 대한 sLLM 미세조정을 시작한다.
한국 법률 지식그래프 시범 개발 착수: 수요가 높은 특정 법률 분야(예: 임대차 분쟁)를 중심으로 NER/RE 도구와 AI Hub의 지식그래프 데이터셋 41을 참조하여 지식그래프 파일럿 개발을 시작한다.
주요 사용자 여정 대상 대화형 AI(메모리 기능 포함) 구현: LangChain 또는 Rasa를 활용하여 핵심적인 사용자 시나리오에 대한 다회성 대화 기능을 구현한다.



3단계: 개인 맞춤형 및 고급 AI 서비스 (24개월 이상)

한국 법률 지식그래프 적용 범위 확대: 더 많은 법률 도메인을 포괄하도록 지식그래프를 확장한다.
지식그래프와 RAG 파이프라인 심층 연동: 고도로 문맥화되고 개인화된 정보 제공을 위해 지식그래프를 RAG 시스템에 깊이 통합한다.
고급 대화형 에이전트 배포: 복잡한 다회성 대화 처리 및 인간 전문가 또는 외부 서비스로의 지능적 라우팅이 가능한 고급 대화형 에이전트를 배포한다.
지속적인 AI 모델 개선 루프 확립: 포괄적인 사용자 피드백 및 성능 모니터링을 기반으로 AI 모델을 지속적으로 개선하는 체계를 구축한다.


이러한 단계적 접근은 공단이 점진적으로 AI 역량을 강화하고, 각 단계마다 실질적인 가치를 창출하며, 장기적인 AI 서비스 혁신 목표를 달성하는 데 도움이 될 것이다. 초기에는 기존 Q&A 시스템을 강화하는 것이 논리적인 출발점이다. 각 단계는 이전 단계의 성과를 기반으로 하며, 예를 들어 1단계에서 강화된 RAG 시스템과 수집된 사용자 피드백은 2단계의 모델 미세조정 및 지식그래프 개발에 중요한 입력 자료가 된다. 3단계에서는 이렇게 축적된 기술과 자산을 바탕으로 진정한 의미의 개인 맞춤형 서비스를 구현하게 된다.B. 법률 AI 시스템을 위한 데이터 거버넌스 및 관리신뢰할 수 있고 정확한 법률 AI 서비스를 구축하기 위해서는 강력한 데이터 거버넌스 체계 확립이 선행되어야 한다. 이는 데이터의 출처 관리부터 품질 관리, 개인정보보호, 버전 관리 등을 포괄하는 종합적인 관리 시스템을 의미한다.
데이터 출처: AI Hub에서 제공하는 방대한 법률 데이터셋 34과 공단 내부에서 축적된 (익명화 처리된) 상담 데이터 5를 주요 데이터 소스로 활용한다.
메타데이터 관리: AI 시스템에 사용되는 모든 법률 문서 및 데이터에 대한 체계적인 메타데이터 관리 시스템을 구현한다.

도구: 오픈소스 옵션인 DataHub 144 또는 Apache Atlas 152를 법률 도메인의 특성에 맞게 커스터마이징하여 활용할 수 있다. 예를 들어, DataHub는 '법령', '판례'와 같은 사용자 정의 개체(entity)와 '관할법원', '주요키워드', '관련조항' 등의 속성(aspect)을 정의할 수 있도록 지원한다.147 Apache Atlas는 '법률명', '판례번호'와 같은 법률 문서의 속성을 포함하는 사용자 정의 유형(type)을 생성하고 분류 체계를 만들 수 있게 한다.155


데이터 품질 및 익명화: 학습 데이터 및 RAG 시스템용 지식베이스의 높은 데이터 품질을 유지해야 한다. 공단 내부 데이터 활용 시에는 개인식별정보(PII)에 대한 철저한 익명화 처리를 통해 개인정보보호 규정을 준수해야 한다.180
버전 관리 및 계보(Lineage) 추적: 데이터셋, AI 모델, 지식그래프 구성요소의 버전을 관리하여 재현성을 보장하고 업데이트를 체계적으로 관리한다. DataHub, Atlas와 같은 도구는 데이터 흐름을 추적하는 계보 관리 기능을 제공할 수 있다.144
보안: 민감한 법률 데이터 및 AI 모델에 대한 안전한 저장 및 접근 통제 시스템을 구축한다.181
법률 데이터는 그 특성상 민감하고 복잡하므로, 적절한 메타데이터(예: 판례 날짜, 관할권, 법령의 위계 등) 관리 없이는 AI의 출력이 부정확하거나 위험할 수 있다. DataHub나 Atlas와 같은 도구는 이러한 복잡성을 체계적으로 관리하고 데이터 품질을 보장하는 데 필수적이다. 특히 공단 내부 데이터의 익명화는 AI 활용 전 반드시 해결해야 할 중요한 윤리적, 법적 과제이다.C. 윤리적 AI 사용 및 사용자 신뢰 확보법률 분야에서 AI를 활용할 때는 기술적 성능만큼이나 윤리적 사용 원칙을 준수하고 사용자 신뢰를 확보하는 것이 매우 중요하다. 공단은 AI 서비스가 공정하고 투명하며 책임감 있는 방식으로 운영될 수 있도록 명확한 가이드라인을 설정하고 이를 실천해야 한다.

법률 정보 제공과 법률 조언의 명확한 구분:

AI 서비스는 법률 정보를 제공하는 역할에 국한되어야 하며, 개인의 구체적인 법적 문제에 대한 해결책을 제시하는 법률 조언을 제공해서는 안 된다. 이는 AI의 현재 기술적 한계와 변호사법 등 관련 법규를 고려한 필수적인 원칙이다.183
모든 AI 생성 콘텐츠에는 해당 정보가 법률 조언이 아니며, 구체적인 법률 문제에 대해서는 반드시 변호사 등 전문가와 상담해야 한다는 명확한 면책 조항(disclaimer)을 포함해야 한다.1



AI 능력과 한계에 대한 투명한 소통:

사용자에게 AI와 상호작용하고 있음을 명확히 알려야 한다.182
AI가 생성하는 정보의 오류 가능성("환각 현상")이나 문맥 이해의 한계 등 AI의 본질적인 제약점을 솔직하게 공개해야 한다.184
AI는 법률 전문가의 업무를 보조하고 효율성을 높이는 도구이지, 전문가를 대체하는 것이 아님을 강조해야 한다.187



데이터 프라이버시 및 보안:

사용자 데이터의 기밀성 유지와 보안에 대해 명확히 안내하고 신뢰를 주어야 한다.180
대한민국의 개인정보보호법 등 관련 법규를 철저히 준수해야 한다.



편향성 완화:

AI 모델과 학습 데이터셋에 존재할 수 있는 편향이 불공정하거나 차별적인 결과를 초래하지 않도록 정기적인 감사와 검증을 수행해야 한다.188



인간 감독 및 개입 보장:

AI 시스템의 개발, 배포, 운영 전반에 걸쳐 적절한 수준의 인간 감독이 이루어져야 한다.185
AI가 만족스러운 답변을 제공하지 못하거나 사용자가 인간과의 상담을 선호할 경우, 원활하게 인간 상담원으로 연결될 수 있는 명확한 경로를 제공해야 한다.186


사용자 신뢰는 법률 AI 서비스의 성공을 위한 가장 중요한 기반이다. 따라서 공단은 AI를 유용한 정보 제공 도구로 포지셔닝하고, 그 능력과 한계를 투명하게 공개하며, 윤리적 원칙을 철저히 준수함으로써 사용자 신뢰를 구축하고 유지해야 한다. AI가 법률 조언의 영역을 침범할 위험은 법률 분야에서 특히 크므로, AI의 페르소나, 프롬프트 설계, 면책 조항 등을 통해 정보 제공과 조언의 경계를 지속적으로 강조해야 한다. AI의 한계를 투명하게 공개하는 것이 과장된 성능을 내세우는 것보다 장기적으로 사용자 신뢰를 얻는 데 더 효과적이다.187D. 지속적 개선 주기 확립: 피드백 수집 및 모델 반복AI 서비스는 한번 구축하고 끝나는 시스템이 아니라, 지속적인 모니터링, 사용자 피드백 수집, 그리고 이를 통한 반복적인 개선 과정을 통해 시간이 지남에 따라 서비스의 관련성, 정확성, 사용자 신뢰도를 유지하고 향상시켜야 한다.

사용자 피드백 메커니즘:

명시적 피드백: AI 답변에 대한 사용자의 직접적인 평가를 수집하기 위해 '좋아요/싫어요', 별점 평가, 간단한 의견 입력창 등의 기능을 구현한다. Langfuse 168는 사용자 피드백(수치형, 범주형, 불리언)을 특정 실행 추적(trace)에 연결하여 수집할 수 있는 기능을 제공한다. Chatwoot 26은 CSAT(고객 만족도) 설문조사 기능을 중심으로 피드백을 수집한다.
암묵적 피드백: 사용자 상호작용 패턴(예: 질의 재구성, 세션 이탈, 추천 자료 클릭률 등)을 분석하여 만족도 및 혼란 지점을 간접적으로 파악한다.168



피드백 분석 및 조치:

수집된 피드백을 정기적으로 검토하여 공통적인 문제점, 정보 부정확성, AI 성능 미흡 영역 등을 식별한다.168
피드백을 바탕으로 프롬프트를 개선하고, RAG 지식베이스를 보강하며(예: 누락된 문서 추가, 분할 방식 개선), 추가적인 미세조정을 위한 데이터를 확보한다.189



모델 평가 및 반복:

자동화된 지표(예: RAG 시스템을 위한 RAGAS)와 인간 검토를 병행하여 AI 모델의 성능을 지속적으로 평가하는 프레임워크를 구축한다.189
새로운 데이터(사용자 피드백으로부터 도출된 수정된 예제 포함)를 활용하여 주기적으로 모델을 재학습하거나 미세조정한다.



A/B 테스트: Langfuse와 같은 플랫폼을 활용하여 다양한 모델 버전이나 프롬프트 전략을 A/B 테스트하고, 사용자 만족도 및 작업 성공률에 미치는 영향을 측정한다.170


후속 조치를 위한 알림 시스템:

AI가 즉각적으로 완전한 해결책을 제공하기 어렵거나 인간의 개입이 예정된 복잡한 질의의 경우, 사용자에게 진행 상황 업데이트, 다음 단계 알림, 또는 준비사항 체크리스트 등을 제공하기 위한 알림 시스템(이메일/SMS)을 구현한다.191
이를 위해 오픈소스 Python 라이브러리(이메일: smtplib, SuprSend SDK 195, EasyGmail 196; SMS: Twilio API 및 Python 헬퍼 라이브러리 197)를 활용할 수 있다.


사용자 피드백은 AI가 사용자의 실제 요구를 충족하는지 이해하는 가장 직접적인 방법이다. Langfuse 168와 같은 도구는 피드백을 특정 AI 상호작용(추적)에 연결하는 강력한 메커니즘을 제공하여, 목표 지향적인 디버깅 및 개선을 가능하게 한다. 예를 들어, 다수의 사용자가 "소액사건 이행권고결정"에 대한 AI의 답변이 혼란스럽다고 지적한다면, 이는 RAG 지식베이스 내 해당 주제의 정보를 수정하거나 LLM의 설명 능력을 미세조정해야 함을 시사한다. 알림 시스템 191은 특히 AI가 즉각적인 해결책을 제공할 수 없을 때 사용자와의 소통을 유지하고 필요한 후속 조치를 안내함으로써 사용자 경험을 개선하는 데 기여한다.V. 결론 및 향후 전망A. 기술적 타당성 및 전략적 중요성 요약본 심층 분석을 통해, 대한민국 법률구조공단의 AI 서비스를 오픈소스 기술 기반으로 개선하는 것은 기술적으로 충분히 타당하다는 결론을 도출하였다. 한국어 거대언어모델(LLM)의 발전, AI Hub를 중심으로 한 풍부한 법률 데이터셋의 가용성, 그리고 LangChain, Rasa와 같은 성숙한 AI 개발 프레임워크의 존재는 이러한 기술적 실현 가능성을 강력하게 뒷받침한다. KoSimCSE와 같은 한국어 임베딩 모델은 검색 증강 생성(RAG) 시스템의 핵심인 정보 검색 단계에서 높은 성능을 기대할 수 있으며, Polyglot-Ko, SOLAR mini 등 다양한 규모와 특성을 가진 한국어 LLM들은 법률 분야 특화 미세조정을 통해 질의응답, 문서 요약, 대화형 에이전트 등 다양한 서비스 구현의 기반이 될 수 있다.이러한 AI 서비스 개선은 단순한 기술 도입을 넘어, 법률 정보 접근성을 획기적으로 향상시키고 공단의 운영 효율성을 증대시키며, 나아가 국민들이 자신의 권리를 인지하고 보호하는 데 필요한 정보를 보다 쉽게 얻을 수 있도록 지원한다는 점에서 전략적으로 매우 중요하다. 사용자 중심의 문제 해결(자가 해결 지원, 외부 도움 연계, 소송 절차 안내) 및 맞춤형 제안 기능은 공단 서비스의 질을 한 차원 높이고, 법률구조 서비스의 패러다임을 변화시킬 잠재력을 지닌다.B. 공공 법률구조 서비스 분야 AI의 장기 비전공공 법률구조 서비스 분야에서 AI의 장기적인 비전은 단순 정보 제공을 넘어, 예방적 법률 지원, 맞춤형 법률 컨설팅, 그리고 효율적인 분쟁 해결 지원으로 확장될 수 있다.
예방적 법률 지원: AI는 잠재적인 법적 문제 발생 가능성을 사전에 인지하고 사용자에게 관련 정보를 제공함으로써 분쟁을 예방하는 역할을 할 수 있다. 예를 들어, 특정 계약 유형에 대한 일반적인 위험 요소를 분석하고 사용자에게 맞춤형 체크리스트를 제공하거나, 변경된 법규에 대한 선제적인 알림 서비스를 제공할 수 있다.
고도화된 개인 맞춤형 컨설팅: 법률 지식그래프와 정교한 대화형 AI의 결합은 사용자의 복잡한 상황을 심층적으로 이해하고, 관련된 법령, 판례, 유사 사례를 종합적으로 분석하여 매우 구체적이고 개인화된 정보(법률 조언이 아닌)를 제공하는 수준으로 발전할 수 있다. 이는 사용자가 자신의 법적 상황을 명확히 이해하고 합리적인 의사 결정을 내리는 데 크게 기여할 것이다.
효율적인 분쟁 해결 지원: 초기 단계의 간단한 분쟁에 대해서는 AI가 대안적 분쟁 해결(ADR) 절차를 안내하거나, 관련 서류 작성 지원, 절차적 정보 제공 등을 통해 사용자가 보다 효율적으로 문제에 대응할 수 있도록 지원할 수 있다. 물론, 모든 과정에서 인간 전문가의 감독과 개입은 필수적이다.
법률 데이터 분석 및 정책 제언: 공단에 축적되는 방대한 법률 상담 데이터와 AI 상호작용 데이터는 익명화 및 분석을 통해 사회적으로 빈번하게 발생하는 법률 문제 유형, 법률 시스템의 개선점 등을 파악하는 데 활용될 수 있으며, 이는 장기적으로 입법 및 정책 개선을 위한 귀중한 기초 자료가 될 수 있다.
LLM 기술과 AI 윤리에 대한 논의가 지속적으로 발전함에 따라, 공공 법률구조 분야에서의 AI 활용 범위와 책임 또한 계속해서 진화할 것이다. 공단은 이러한 기술적, 사회적 변화에 능동적으로 대응하며, AI를 통해 모든 국민이 보다 쉽게 법의 보호를 받을 수 있는 환경을 조성하는 데 선도적인 역할을 수행해야 할 것이다. 이를 위해서는 기술 개발과 함께 윤리적 지침 수립, 데이터 보안 강화, 그리고 사용자 교육에도 지속적인 투자가 이루어져야 한다.제안된 개선 아이템 및 오픈소스 구현 전략 요약표다음 표는 본 보고서에서 논의된 주요 개선 아이템과 각 아이템을 오픈소스 기술로 구현하기 위한 핵심 전략을 요약한 것이다.
개선 아이템 (사용자 요구)제안 오픈소스 기술/접근 방식핵심 오픈소스 구성요소 (예시)주요 기술적 고려사항/도전 과제사용자/공단 주요 효익향상된 법률 Q&ALangChain RAG 파이프라인 구축, 한국어 임베딩 모델 및 미세조정된 한국어 sLLM 활용KoSimCSE (임베딩), Polyglot-Ko 1.3B/SOLAR mini (생성형 LLM, 미세조정), AI Hub 법률 Q&A 데이터셋, ChromaDB/FAISS (벡터DB), LangChain법률 용어 및 문맥 이해를 위한 LLM 미세조정의 복잡성, 방대한 법률 문서의 효과적인 분할(chunking) 및 인덱싱, 답변의 사실적 정확성 및 최신성 유지, 환각 현상 방지사용자의 법률 질문에 대한 빠르고 정확하며 신뢰할 수 있는 정보 획득, 공단 상담원의 단순 반복 문의 응대 부담 감소소액사건심판 절차 안내Rasa 또는 LangChain Agent 기반 대화형 AI, 단계별 절차 안내 로직 구현, 관련 법령/서식 정보 연동Rasa/LangChain Agents, AI Hub 법률 지식베이스 (소액사건 관련 법령, 절차 정보), 공단 제공 소액사건 안내자료절차 변경에 따른 정보 업데이트의 신속성, 사용자의 다양한 상황 변수를 고려한 맞춤형 안내의 어려움, 법률 정보와 법률 조언 간의 명확한 경계 유지사용자가 소액사건 절차를 쉽게 이해하고 스스로 준비할 수 있도록 지원, 절차적 실수 최소화, 공단의 안내 업무 효율화개인 맞춤형 법률 정보 제공한국 법률 지식그래프(KG) 구축 및 활용, KG 기반 RAG 고도화, 다회성 대화 관리 및 문맥 이해 강화KRLawGPT 데이터(접근 가능 시) 또는 AI Hub 지식그래프 데이터41 기반 KG 구축, 오픈소스 NER/RE 도구(미세조정), Neo4j(그래프DB), LangChain (메모리 기능, 고급 프롬프팅)대규모 한국 법률 지식그래프 구축 및 유지보수의 높은 비용과 전문성 요구, 법률 개념 및 관계 정의의 복잡성, KG와 LLM 간의 효과적인 연동 및 추론 능력 확보사용자의 구체적인 상황과 관련된 법률 정보, 판례, 유사 사례 등을 종합적으로 제공받아 문제 해결 능력 향상, 공단 서비스 만족도 제고인간 상담원/외부기관 지능적 연계Rasa/LangChain Agent 기반 초기 진단 및 라우팅 시스템, 공단 내부 상담 시스템 및 외부 기관 정보 연동Rasa/LangChain Agents, 공단 상담 예약 시스템 API(필요시 개발), 외부기관 정보 DB다양한 법률 문제 유형에 대한 정확한 초기 진단 로직 개발, 공단 내부 및 외부 기관 정보의 최신성 유지 및 연동의 기술적 복잡성, 개인정보보호 및 정보 공유 관련 규정 준수사용자가 자신의 문제에 가장 적합한 지원 채널(공단 상담원, 특정 부서, 외부 전문기관 등)로 신속하고 정확하게 안내받아 문제 해결 시간 단축, 공단 자원의 효율적 배분법률 문서 자동 요약한국어 sLLM 기반 요약 모델 미세조정, AI Hub 법률 문서 요약 데이터셋 활용Polyglot-Ko/SOLAR mini (미세조정), AI Hub 문서 요약 데이터셋 38법률 문서의 핵심 내용을 정확하게 추출하고 간결하게 생성하는 능력 확보, 원문의 법적 뉘앙스 보존, 다양한 유형의 법률 문서(판결문, 법령, 계약서 등)에 대한 일반화 성능 확보사용자가 길고 복잡한 법률 문서를 빠르게 이해할 수 있도록 지원, 공단 내부적으로 문서 검토 및 분석 시간 단축사용자 피드백 기반 지속적 개선Langfuse 또는 Chatwoot과 같은 오픈소스 피드백 수집 도구 도입, 피드백 분석 및 모델/데이터베이스 업데이트 파이프라인 구축, 알림 시스템(이메일, SMS) 연동Langfuse, Chatwoot, Python 이메일/SMS 라이브러리(smtplib, Twilio SDK 등)사용자 피드백 수집률 제고 방안, 수집된 피드백의 정성적/정량적 분석 및 실제 개선으로 이어지는 체계 구축, 피드백 반영 주기의 적절성, 알림 시스템의 개인정보보호 및 스팸 방지AI 서비스의 정확성, 유용성, 사용자 만족도를 지속적으로 향상시키고, 사용자와의 신뢰 관계 구축. 문제 해결 지연 시 후속 조치 안내를 통해 사용자 경험 개선.
