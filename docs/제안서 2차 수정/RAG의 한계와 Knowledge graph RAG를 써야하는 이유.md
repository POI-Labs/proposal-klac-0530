# RAG의 한계와 Knowledge Graph RAG를 써야하는 이유

## RAG(Retrieval-Augmented Generation)의 주요 한계점

### 1. 문장 유사도 기반 검색의 한계

기존 RAG 시스템은 주로 의미론적 유사성에만 의존하여 검색을 수행하는데, 이는 여러 문제점을 야기합니다[1]. 정확히 일치하는 값보다는 가장 비슷한 값을 가져오기 때문에 유사한 값 검색 시 불필요한 정보가 포함될 수 있습니다[1]. 이러한 작은 불일치가 나비효과처럼 답변 전체의 맥락에 큰 차이를 초래할 수 있습니다[1].

또한 기존 RAG는 단순한 검색과 생성의 반복으로 인해 불필요한 데이터 중복과 노이즈가 발생할 수 있으며, 의미론적 유사성을 고려하지 않은 검색 방법은 실제로 사용자가 원하는 정보와 관련이 없는 결과를 제공하는 경우가 많습니다[2].

### 2. 정보 연결 및 통합의 어려움

RAG는 서로 다른 정보 조각을 연결하여 새로운 통찰을 도출하는 데 어려움을 겪습니다[23]. 하나의 질문에 답변하기 위해 분산된 정보들 사이의 공통 속성을 찾아내고 이를 통합해야 하는 상황에서 기존 RAG는 성능이 떨어집니다[23]. 예를 들어, RAG는 대규모 언어 모델(LLM)이 생성한 응답에서 환각 문제를 해결할 수 있지만, 청크 간의 내재적 관계를 무시하는 한계가 있습니다[26].

### 3. 비정형 데이터 처리의 한계

RAG는 주로 벡터DB에 의존하여 비정형 데이터를 다루는데, 벡터DB는 기존의 정형 데이터를 포함한 관계형DB만큼 효과적으로 비정형 데이터를 처리하지 못한다는 문제점이 있습니다[1]. 하나의 정보를 여러 개의 chunk로 나누어 저장하면서 하나의 문맥적 의미가 여러 chunk에 분리되거나, 서로 다른 chunk에 포함된 개념 간의 관계를 효과적으로 반영하지 못한다는 한계가 있습니다[15].

### 4. 실시간 업데이트의 어려움

RAG 시스템은 외부의 지식 베이스인 벡터DB를 참조하여 답변을 생성하는데, 이 지식 베이스는 주기적으로 업데이트되어야 합니다[1]. 그러나 실제로 벡터DB를 실시간으로 업데이트하는 것에는 기술적, 운영적 어려움이 따라 최신 정보를 반영하지 못하는 기존의 RAG가 생성하는 답변이 오래되거나 부정확할 수 있습니다[1].

### 5. 대규모 데이터셋 분석의 한계

기존 RAG는 대규모 데이터 집합이나 단일 대규모 문서를 요약하고 의미론적 개념을 전체적으로 이해하는 질문에 대해 만족스러운 결과를 내지 못합니다[23]. 예를 들어, "데이터에서 상위 5개의 주제는 무엇인가요?"와 같은 질문은 기본 RAG가 데이터셋 내에서 의미적으로 유사한 텍스트 콘텐츠에 대해 벡터 검색을 수행하기 때문에 제대로 작동하지 않습니다[19].

## Knowledge Graph RAG를 써야하는 이유

### 1. 구조화된 지식 표현과 관계 모델링

Knowledge Graph RAG는 핵심 개념 간의 복잡한 관계를 유연하게 정의하고, 데이터 간 연결 관계를 정보 탐색에 함께 반영함으로써 기존 RAG의 한계에 대응할 수 있습니다[15]. Knowledge Graph는 서로 연관된 정보를 연결하고, 그 속에서 의미 있는 답을 도출해 내는 기술로, 보다 정밀하고 문맥에 맞는 응답을 생성하는 그래프 RAG로 확장되고 있습니다[11].

### 2. 향상된 추론 능력과 컨텍스트 이해

Knowledge Graph는 지식의 구조화 및 표현을 통해 실제 세계의 정보를 구조화된 방식으로 표현하며, 이는 RAG 과정에서 LLM이 이해하고 활용할 수 있는 형태로 지식을 제공합니다[13]. 지식 그래프의 상호 연결된 특성을 활용하여 원시 텍스트 데이터만으로는 어렵거나 불가능한 추론을 도출하고 복잡한 관계를 추론할 수 있습니다[24].

### 3. 정보 연결 및 통합 능력

GraphRAG는 지식 그래프를 활용하여 쿼리에 포함된 엔터티를 인식하고, 관련 정보를 연결하여 더욱 풍부한 답변을 제공합니다[23]. 예를 들어, "노보로시야(Novorossiya)는 무엇을 했는가?"라는 질문에서 기존 RAG가 답을 찾지 못한 반면, GraphRAG는 지식 그래프를 활용해 노보로시야의 활동과 관련된 다양한 사건을 연결하여 구체적인 답변을 생성할 수 있었습니다[23].

### 4. 전체 데이터셋의 의미론적 분석

GraphRAG는 지식 그래프의 클러스터링 기능을 통해 데이터셋 전체의 주요 주제를 효과적으로 요약할 수 있습니다[23]. 이를 통해 "데이터에서 가장 중요한 다섯 가지 주제는 무엇인가?"와 같은 질문에 대해, GraphRAG는 의미론적 클러스터를 바탕으로 명확하고 구체적인 답변을 제공합니다[23].

### 5. 향상된 검색 성능과 정확도

KG2RAG는 의미 기반 검색을 수행하여 초기 청크를 제공한 후, KG 기반 청크 확장 프로세스와 KG 기반 청크 조직화 프로세스를 적용하여 관련성이 높고 중요한 지식을 잘 구성된 단락 형태로 전달합니다[26]. 실험 결과 KG2RAG는 기존 RAG 기반 방법보다 우수한 응답 품질과 검색 품질을 보입니다[26].

### 6. 도메인 특화 성능

Graph RAG는 도메인별 온톨로지와 분류법을 통합하여 특정 도메인에 맞춘 정확한 검색 및 이해를 가능하게 합니다[24]. 이러한 도메인 적응성으로 인해 의료, 금융, 엔지니어링 등 다양한 전문 도메인에서 활용되고 있으며, 높은 도메인 적응성을 발휘하여 스마트하고 효율적인 검색 능력을 제공합니다[24].

## Knowledge Graph RAG의 구현 방식

### Local Search와 Global Search

GraphRAG는 두 가지 주요 검색 방식을 제공합니다[19]. Local Search는 AI가 추출한 지식그래프의 관련 데이터를 원본 문서의 텍스트 조각과 결합하여 답변을 생성하며, 문서에 언급된 특정 엔티티에 대한 이해가 필요한 질문에 적합합니다[19]. Global Search는 AI가 생성한 모든 커뮤니티 보고서를 map-reduce 방식으로 검색하여 답변을 생성하며, 데이터셋 전체에 대한 이해가 필요한 질문에 대해 좋은 응답을 제공합니다[19].

### Microsoft GraphRAG 구현

Microsoft는 GraphRAG를 개발해 오픈소스로 공개했으며, 이는 구조화된, 계층적 접근 방식을 통해 기존의 단순한 의미 검색 방식을 개선했습니다[20]. GraphRAG 프로세스는 원시 텍스트에서 지식 그래프를 추출하고, 커뮤니티 계층을 구축하며, 이러한 커뮤니티에 대한 요약을 생성한 다음, RAG 기반 작업을 수행할 때 이러한 구조를 활용하는 과정을 포함합니다[20].

## 결론

기존 RAG의 문장 유사도 기반 검색, 정보 연결의 어려움, 실시간 업데이트 한계 등의 문제점을 Knowledge Graph RAG가 효과적으로 해결할 수 있습니다[1][15][23]. Knowledge Graph RAG는 구조화된 지식 표현, 향상된 추론 능력, 정보 연결 및 통합, 전체 데이터셋의 의미론적 분석 등의 장점을 통해 더욱 정확하고 맥락에 맞는 응답을 제공할 수 있습니다[13][23][24]. 특히 복잡한 관계와 구조를 표현하고, LLM을 기반으로 정확하고 상황에 맞는 정보를 제공하는 능력을 통해 차세대 RAG 시스템의 핵심 기술로 자리잡고 있습니다[11][24].

[1] https://blog-ko.superb-ai.com/limitations-and-workarounds-for-rag/
[2] https://digitalbourgeois.tistory.com/410
[3] http://www.apple-economy.com/news/articleView.html?idxno=74922
[4] https://www.stephendiehl.com/posts/rag/
[5] https://aristudy.tistory.com/126
[6] https://edbkorea.com/blog/llm%EC%9D%98-%ED%95%9C%EA%B3%84%EC%99%80-rag%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0/
[7] https://arxiv.org/html/2401.05856v1
[8] https://arxiv.org/abs/2502.06864
[9] https://www.datacamp.com/tutorial/knowledge-graph-rag
[10] https://blog.naver.com/imkyungwon/223504747257
[11] https://www.elancer.co.kr/blog/detail/841
[12] https://devocean.sk.com/community/detail.do?ID=166239&boardType=DEVOCEAN_STUDY&page=1
[13] https://shadowego.com/articles/745
[14] https://uoahvu.tistory.com/entry/GraphRAG-Neo4j-DB%EC%99%80-LangChain-%EA%B2%B0%ED%95%A9%EC%9D%84-%ED%86%B5%ED%95%9C-%EC%A7%88%EC%9D%98%EC%9D%91%EB%8B%B5-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-Kaggle-CSV-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%81%EC%9A%A9%ED%95%98%EA%B8%B0
[15] https://aws.amazon.com/ko/blogs/tech/amazon-bedrock-graph-rag/
[16] https://zdnet.co.kr/view/?no=20250426193245
[17] https://neo4j.com/blog/developer/microsoft-graphrag-neo4j/
[18] https://www.linkedin.com/pulse/vector-databases-vs-knowledge-graphs-choosing-right-harsha-srivatsa-a1hic
[19] https://chaechaecaley.tistory.com/23?category=1248230
[20] https://microsoft.github.io/graphrag/
[21] https://writer.com/engineering/vector-database-vs-graph-database/
[22] https://www.themoonlight.io/ko/review/rag-vs-graphrag-a-systematic-evaluation-and-key-insights
[23] https://digitalbourgeois.tistory.com/543
[24] https://seo.goover.ai/report/202407/go-public-report-ko-a1361160-7c6f-4039-a7c3-6f3956209b17-0-0.html
[25] https://turingpost.co.kr/p/topic-22-advanced-rag
[26] https://jik9210.tistory.com/75
[27] https://aiheroes.ai/community/276
[28] https://www.igloo.co.kr/security-information/ragretrieval-augmented-generation-llm%EC%9D%98-%ED%95%9C%EA%B3%84%EC%99%80-%EB%B3%B4%EC%99%84-%EB%B0%A9%EB%B2%95/
[29] https://modulabs.co.kr/blog/retrieval-augmented-generation
[30] https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/
[31] https://brunch.co.kr/@@gDYF/38
[32] https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_rag_query_engine/
[33] https://neo4j.com/blog/developer/knowledge-graph-rag-application/
[34] https://kr.linkedin.com/posts/aldente0630_graph-rag%EC%9D%98-%EB%AA%A8%EB%93%A0-%EA%B2%83-activity-7319962816952553472-nykQ
[35] https://bitnine.tistory.com/585
[36] https://velog.io/@looa0807/GraphRAG-%EC%9C%A0%ED%8A%9C%EB%B8%8C-%EC%98%81%EC%83%81-%EC%A0%95%EB%A6%AC
[37] https://selectstar.ai/blog/tech/graphrag-1-framework/
[38] https://lilys.ai/notes/961215