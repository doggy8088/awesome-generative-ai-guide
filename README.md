# :star: :bookmark: awesome-generative-ai-guide

生成式 AI 正在迅速增長，本資料庫是關於生成式 AI 研究、面試材料、筆記本等更新的綜合中心！

探索以下資源:

1. [每月最佳 GenAI 論文列表](#star-最佳生成-ai-論文列表-2024年6月)
2. [GenAI 面試資源](#computer-面試準備)
3. [應用 LLMs 精通 2024 (由 Aishwarya Naresh Reganti 建立) 課程材料](#mortar_board-課程)
4. [所有 GenAI 相關的免費課程列表 (列出了超過 85 個)](#book-免費生成式-ai-課程列表)
5. [開發生成式 AI 應用程式的程式碼庫/筆記本列表](#notebook-程式碼筆記本)

我們將定期更新此儲存庫，請隨時關注最新的新增內容！

快樂學習！

---


## [2024年6月] 🔥🔥 現在開放 "生成式 AI 天才" 的註冊

- 基於短片/短影片的20天免費入門課程（不需要AI背景！）
- 查看更多資訊並註冊[此處](free_courses/generative_ai_genius/README.md)。
- 盡快註冊！課程將於2024年7月8日開始！

---


## :speaker: 公告

- Applied LLMs Mastery 全課程內容已發布!!! ([點擊這裡](free_courses/Applied_LLMs_Mastery_2024))
- 5天學習LLM基礎的路線圖現已推出! ([點擊這裡](resources/genai_roadmap.md))
- 60個常見的GenAI面試問題現已推出! ([點擊這裡](interview_prep/60_gen_ai_questions.md))
- ICLR 2024論文摘要 ([點擊這裡](https://areganti.notion.site/06f0d4fe46a94d62bff2ae001cfec22c?v=d501ca62e4b745768385d698f173ae14))
- 免費GenAI課程列表 ([點擊這裡](https://github.com/aishwaryanr/awesome-generative-ai-guide#book-list-of-free-genai-courses))
- 生成式AI資源和路線圖
  - [3天RAG路線圖](resources/RAG_roadmap.md)
  - [5天LLM基礎路線圖](resources/genai_roadmap.md)
  - [5天LLM代理路線圖](resources/agents_roadmap.md)
  - [代理101指南](resources/agents_101_guide.md)
  - [MM LLMs簡介](resources/mm_llms_guide.md)
  - [LLM術語系列: 常用LLM術語及其易懂的定義](resources/llm_lingo)

---


## :star: 最佳生成 AI 論文列表 (2024年6月)

\*每月末更新
| 日期 | 標題 | 摘要 | 主題 |
|------|-------|----------|--------|
| 2024年6月28日 | [Step-DPO: 用於LLMs長鏈推理的逐步偏好最佳化](https://arxiv.org/abs/2406.18629) | 數學推理對大型語言模型（LLMs）提出了重大挑戰，因為準確性需要廣泛且精確的推理鏈。確保每一步推理的正確性至關重要。為了解決這個問題，我們旨在通過學習人類反饋來增強LLMs的穩健性和事實性。然而，直接偏好最佳化（DPO）對於長鏈數學推理顯示出有限的好處，因為使用DPO的模型難以識別錯誤答案中的詳細錯誤。這一限制源於缺乏細粒度的過程監督。我們提出了一種簡單、有效且數據高效的方法，稱為Step-DPO，將個別推理步驟作為偏好最佳化的單位，而不是整體評估答案。此外，我們開發了一個數據構建管道，用於Step-DPO，從而創建包含10K逐步偏好對的高品質數據集。我們還觀察到，在DPO中，自生成數據比人類或GPT-4生成的數據更有效，因為後者具有分布外的特性。我們的研究結果顯示，僅需10K偏好數據對和少於500個Step-DPO訓練步驟，就能在具有超過70B參數的模型上在MATH上獲得近3%的準確性提升。值得注意的是，當Step-DPO應用於Qwen2-72B-Instruct時，在MATH和GSM8K的測試集上分別達到70.8%和94.0%的得分，超越了一系列閉源模型，包括GPT-4-1106、Claude-3-Opus和Gemini-1.5-Pro。 | 數學推理, 最佳化 |
| 2024年6月28日 | [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094) | 我們提出了一種新穎的以人物為驅動的數據合成方法，利用大型語言模型（LLM）內的各種視角來創建多樣的合成數據。為了充分利用這種方法，我們引入了Persona Hub——一個從網絡數據中自動策劃的10億多樣化人物集合。這10億人物（約佔世界總人口的13%），作為世界知識的分佈式載體，可以觸及LLM內幾乎所有的視角，從而促進各種場景下大規模多樣化合成數據的創建。通過展示Persona Hub在大規模合成高品質數學和邏輯推理問題、指令（即用戶提示）、知識豐富的文本、遊戲NPC和工具（函式）方面的應用，我們展示了以人物為驅動的數據合成具有多功能性、可擴展性、靈活性和易用性，可能推動合成數據創建和應用的範式轉變，對LLM的研究和開發產生深遠影響。 | 合成數據生成 |
| 2024年6月27日 | [WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models](https://arxiv.org/abs/2406.18510) | 我們介紹了WildTeaming，一個自動LLM安全紅隊框架，通過挖掘野外用戶-聊天機器人互動來發現5.7K個獨特的新的越獄策略集群，然後組合多種策略以系統地探索新的越獄方法。與之前通過招募人類工作者、基於梯度的最佳化或與LLMs的迭代修訂進行紅隊測試的工作相比，我們的工作調查了未被特別指示破壞系統的聊天機器人用戶的越獄行為。WildTeaming揭示了前沿LLMs以前未識別的漏洞，與最先進的越獄方法相比，導致多達4.6倍更多樣且成功的對抗性攻擊。雖然存在許多越獄評估數據集，但很少有開放原始碼的越獄訓練數據集，因為即使模型權重是開放的，安全訓練數據也被封閉。通過WildTeaming，我們創建了WildJailbreak，一個大規模的開放原始碼合成安全數據集，包含262K個普通（直接請求）和對抗性（複雜越獄）提示-回應對。為了減輕誇大的安全行為，WildJailbreak提供了兩種類比查詢：1）有害查詢（普通和對抗性）和2）形式上類似於有害查詢但不含有害內容的良性查詢。由於WildJailbreak大大提升了現有安全資源的品質和規模，它獨特地使我們能夠在安全訓練期間檢查數據的擴展效應以及數據屬性和模型能力之間的相互作用。通過廣泛的實驗，我們確定了實現理想安全行為的訓練屬性：適當的保護而不過度拒絕，有效處理普通和對抗性查詢，並且一般能力的減少最小甚至沒有。WildJailbreak的所有組成部分都有助於實現模型的平衡安全行為。 | 紅隊測試, LLM攻擊 |
| 2024年6月27日 | [LiveBench: A Challenging, Contamination-Free LLM Benchmark](https://arxiv.org/abs/2406.19314) | 測試集污染，即基準測試數據進入新模型的訓練集，是公平LLM評估的眾所周知的障礙，並且可以迅速使基準測試過時。為了減輕這一問題，許多最近的基準測試從人類或LLM評審中眾包新的提示和評估；然而，這些方法可能引入顯著的偏見，並且在評分難題時會崩潰。在這項工作中，我們引入了一個新的LLM基準測試，旨在免受測試集污染和LLM評審和人類眾包的陷阱。我們發布了LiveBench，這是第一個（1）包含來自最新資訊來源的經常更新的問題，（2）根據客觀的真實值自動評分答案，（3）包含各種挑戰性任務，涵蓋數學、編碼、推理、語言、指令遵循和數據分析的基準測試。為了實現這一目標，LiveBench包含基於最近發布的數學競賽、arXiv論文、新聞文章和數據集的問題，並且包含來自以前基準測試如Big-Bench Hard、AMPS和IFEval的任務的更難、無污染版本。我們評估了許多著名的閉源模型，以及從0.5B到110B大小的數十個開放原始碼模型。LiveBench很難，頂級模型的準確率低於65%。我們發布了所有問題、程式碼和模型答案。問題將每月添加和更新，我們將隨著時間的推移發布新的任務和更難的任務版本，以便LiveBench能夠區分LLMs隨著未來改進的能力。我們歡迎社區參與和合作，以擴展基準測試任務和模型。 | 基準測試, 數據集 |
| 2024年6月26日 | [Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation](https://arxiv.org/abs/2406.18676) | 檢索增強生成（RAG）在減輕大型語言模型（LLMs）的幻覺問題方面顯示了效果。然而，使檢索器與多樣化的LLMs的知識偏好對齊的困難不可避免地對開發可靠的RAG系統構成挑戰。為了解決這個問題，我們提出了DPA-RAG，一個旨在對齊RAG系統內多樣知識偏好的通用框架。具體來說，我們首先引入了一個偏好知識構建管道，並結合了五種新穎的查詢增強策略，以緩解偏好數據的稀缺性。基於偏好數據，DPA-RAG實現了外部和內部偏好對齊：1）它將成對、點對和對比偏好對齊能力整合到重排器中，實現RAG組件之間的外部偏好對齊。2）它進一步在普通監督微調（SFT）之前引入了一個預對齊階段，使LLMs能夠隱式捕捉與其推理偏好對齊的知識，實現LLMs的內部對齊。跨四個知識密集型QA數據集的實驗結果表明，DPA-RAG優於所有基線，並無縫整合了黑盒和開放原始碼的LLM讀者。進一步的定性分析和討論也為實現可靠的RAG系統提供了經驗指導。 | RAG, 對齊 |
| 2024年6月21日 | [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/abs/2406.15319) | 在傳統的RAG框架中，基本的檢索單元通常很短。常見的檢索器如DPR通常與100字的維基百科段落一起工作。這樣的設計迫使檢索器在大語料庫中搜索「針」單元。相比之下，讀者只需要從短的檢索單元中提取答案。這樣的不平衡「重」檢索器和「輕」讀者設計可能導致次優性能。為了緩解這種不平衡，我們提出了一個新框架LongRAG，包括一個「長檢索器」和一個「長讀者」。LongRAG將整個維基百科處理成4K-token單元，比以前長30倍。通過增加單元大小，我們顯著減少了總單元數量，從22M減少到700K。這顯著降低了檢索器的負擔，導致顯著的檢索得分：NQ上的答案召回@1=71%（以前為52%）和HotpotQA（全維基）上的答案召回@2=72%（以前為47%）。然後我們將前k個檢索單元（約30K tokens）餵給現有的長上下文LLM進行零樣本答案提取。無需任何訓練，LongRAG在NQ上達到62.7%的EM，這是已知的最佳結果。LongRAG在HotpotQA（全維基）上也達到64.3%，與SoTA模型相當。我們的研究為將RAG與長上下文LLMs結合的未來路線圖提供了見解。 | RAG |
| 2024年6月20日 | [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) | 今天，我們推出了Claude 3.5 Sonnet——我們即將推出的Claude 3.5模型家族的首次發布。Claude 3.5 Sonnet提高了行業的智能標準，在廣泛的評估中表現優於競爭對手模型和Claude 3 Opus，並且具有我們中端模型Claude 3 Sonnet的速度和成本。 | 基礎LLM |
| 2024年6月20日 | [Can LLMs Learn by Teaching? A Preliminary Study](https://arxiv.org/abs/2406.14629) | 教學以改進學生模型（例如知識蒸餾）是LLMs中廣泛研究的方法。然而，對於人類來說，教學不僅能提高學生的能力，還能提高教師的能力。我們問：LLMs也能通過教學（LbT）學習嗎？如果是，我們可以潛在地解鎖不僅依賴於人類生成數據或更強模型的可能性，持續推進模型。在本文中，我們對這一雄心勃勃的議程進行了初步探索。我們展示了LbT的想法可以融入現有的LLM訓練/提示管道，並提供顯著改進。具體來說，我們設計了三種方法，每種方法模仿人類LbT的三個層次之一：觀察學生的反饋，從反饋中學習，並迭代學習，目標是提高答案準確性而不進行訓練，並通過微調提高模型的內在能力。研究結果令人鼓舞。例如，類似於人類的LbT，我們看到：1）LbT可以引發弱到強的泛化：強模型可以通過教其他弱模型來提高自己；2）學生的多樣性可能有幫助：教多個學生可能比教一個學生或教師本身更好。我們希望這一早期的承諾能激發未來對LbT的研究，並更廣泛地採用教育中的先進技術來改進LLMs。程式碼可在https://github.com/imagination-research/lbt獲得。 | LLM學習 |
| 2024年6月19日 | [Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?](https://arxiv.org/abs/2406.13121) | 長上下文語言模型（LCLMs）有潛力徹底改變我們對傳統依賴外部工具如檢索系統或數據庫的任務的處理方式。利用LCLMs本地攝取和處理整個信息語料庫的能力提供了許多優勢。它通過消除對工具的專業知識的需求提高了用戶友好性，提供了穩健的端到端建模，最小化了複雜管道中的級聯錯誤，並允許在整個系統中應用複雜的提示技術。為了評估這一範式轉變，我們引入了LOFT，一個需要上下文達到數百萬tokens的實際任務基準，用於評估LCLMs在上下文檢索和推理方面的性能。我們的研究結果顯示，LCLMs在未經專門訓練的情況下，能夠與最先進的檢索和RAG系統競爭。然而，LCLMs在SQL類任務所需的組合推理等領域仍面臨挑戰。值得注意的是，提示策略顯著影響性能，強調隨著上下文長度的增加，需要持續研究。總體而言，LOFT為LCLMs提供了一個嚴格的測試平台，展示了它們在模型能力擴展時取代現有範式和處理新任務的潛力。 | 長上下文, 分析 |
| 2024年6月18日 | [Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges](https://arxiv.org/abs/2406.12624) | 提供一個有前途的解決方案來應對與人類評估相關的可擴展性挑戰，LLM-as-a-judge範式迅速獲得了作為評估大型語言模型（LLMs）的方法的吸引力。然而，關於這一範式的優勢和劣勢以及可能存在的偏見，仍有許多未解決的問題。在本文中，我們對充當評審的各種LLMs的性能進行了全面研究。我們利用TriviaQA作為評估LLMs客觀知識推理的基準，並將它們與我們發現具有高互評一致性的人類註釋進行比較。我們的研究包括9個評審模型和9個考試模型——包括基礎模型和指令調整模型。我們評估了不同模型大小、家族和評審提示下的評審模型的對齊情況。在其他結果中，我們的研究重新發現了使用Cohen's kappa作為對齊度量而不是簡單百分比一致性的重要性，顯示出具有高百分比一致性的評審仍然可以分配截然不同的分數。我們發現Llama-3 70B和GPT-4 Turbo與人類的對齊度非常好，但在排名考試模型方面，它們被JudgeLM-7B和詞彙評審Contains所超越，後

## :mortar_board: 課程

#### [進行中] 應用LLM精通2024

加入超過1000名學生的行列，在這10週的冒險中，我們將深入探討LLM在各種使用案例中的應用

#### [連結](https://areganti.notion.site/Applied-LLMs-Mastery-2024-562ddaa27791463e9a1286199325045c)到課程網站。

##### [2024年2月] 註冊仍然開放[點擊這裡](https://forms.gle/353sQMRvS951jDYu7)進行註冊。

🗓️\*第 1 週 [2024 年 1 月 15 日]**\*: [實用的 LLMs 介紹](free_courses/Applied_LLMs_Mastery_2024/week1_part1_foundations.md)**

- 應用 LLM 基礎
- 真實世界 LLM 使用案例
- 領域和任務適應方法

🗓️\*第2週 [2024年1月22日]**\*: [提示與提示工程](free_courses/Applied_LLMs_Mastery_2024/week2_prompting.md)**

- 基本提示原則
- 提示類型
- 應用、風險和進階提示

🗓️\*第3週 [2024年1月29日]**\*: [LLM 微調](free_courses/Applied_LLMs_Mastery_2024/week3_finetuning_llms.md)**

- 微調的基礎
- 微調的類型
- 微調的挑戰

🗓️\*第 4 週 [2024 年 2 月 5 日]**\*: [RAG (檢索增強生成)](free_courses/Applied_LLMs_Mastery_2024/week4_RAG.md)**

- 了解LLM中RAG的概念
- RAG的關鍵組成部分
- 高級RAG方法

🗓️\*第5週 [2024年2月12日]**\*: [建構LLM應用程式的工具](free_courses/Applied_LLMs_Mastery_2024/week5_tools_for_LLM_apps.md)**

- 微調工具
- RAG 工具
- 用於可觀察性、提示、服務、向量搜尋等的工具

🗓️\*第6週 [2024年2月19日]**\*: [評估技術](free_courses/Applied_LLMs_Mastery_2024/week6_llm_evaluation.md)**

- 評估類型
- 常見評估基準
- 常見指標

🗓️\*第7週 [2024年2月26日]**\*: [建構你自己的 LLM 應用程式](free_courses/Applied_LLMs_Mastery_2024/week7_build_llm_app.md)**

- LLM 應用程式的組件
- 從頭到尾建構你自己的 LLM 應用程式

🗓️\*第8週 [2024年3月4日]\*: [進階功能和部署](free_courses/Applied_LLMs_Mastery_2024/week8_advanced_features.md)

- LLM 生命週期和 LLMOps
- LLM 監控和可觀察性
- 部署策略

🗓️\*第9週 [2024年3月11日]**\*: [LLM的挑戰](free_courses/Applied_LLMs_Mastery_2024/week9_challenges_with_llms.md)**

- 延展性挑戰
- 行為挑戰
- 未來方向

🗓️\*第10週 [2024年3月18日]**\*: [新興研究趨勢](free_courses/Applied_LLMs_Mastery_2024/week10_research_trends.md)**

- 更小且效能更佳的模型
- 多模態模型
- 大型語言模型對齊

🗓️*第11週 *獎勵\* [2024年3月25日]**\*: [基礎](free_courses/Applied_LLMs_Mastery_2024/week11_foundations.md)**

- 生成模型基礎
- 自注意力和Transformer
- 語言神經網絡

---


#### :book: 免費生成式 AI 課程列表

##### LLM 基礎與基礎原理

1. [大型語言模型](https://rycolab.io/classes/llm-s23/) by ETH Zurich

2. [理解大型語言模型](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/) by Princeton

3. [Transformers 課程](https://huggingface.co/learn/nlp-course/chapter1/1) by Huggingface

4. [NLP 課程](https://huggingface.co/learn/nlp-course/chapter1/1) by Huggingface

5. [CS324 - 大型語言模型](https://stanford-cs324.github.io/winter2022/) by Stanford

6. [使用大型語言模型的生成式 AI](https://www.coursera.org/learn/generative-ai-with-llms) by Coursera

7. [生成式 AI 入門](https://www.coursera.org/learn/introduction-to-generative-ai) by Coursera

8. [生成式 AI 基礎](https://www.cloudskillsboost.google/paths/118/course_templates/556) by Google Cloud

9. [大型語言模型入門](https://www.cloudskillsboost.google/paths/118/course_templates/539) by Google Cloud
10. [生成式 AI 入門](https://www.cloudskillsboost.google/paths/118/course_templates/536) by Google Cloud
11. [生成式 AI 概念](https://www.datacamp.com/courses/generative-ai-concepts) by DataCamp (Daniel Tedesco Data Lead @ Google)
12. [1 小時大型語言模型入門](https://www.youtube.com/watch?v=xu5_kka-suc) by WeCloudData
13. [從零開始的大型語言模型基礎 | 入門](https://www.youtube.com/watch?v=W0c7jQezTDw&list=PLTPXxbhUt-YWjMCDahwdVye8HW69p5NYS) by Databricks
14. [生成式 AI 解釋](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-07+V1/) by Nvidia
15. [Transformer 模型和 BERT 模型](https://www.cloudskillsboost.google/course_templates/538) by Google Cloud
16. [決策者的生成式 AI 學習計劃](https://explore.skillbuilder.aws/learn/public/learning_plan/view/1909/generative-ai-learning-plan-for-decision-makers) by AWS
17. [負責任的 AI 入門](https://www.cloudskillsboost.google/course_templates/554) by Google Cloud
18. [生成式 AI 基礎](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/) by Microsoft Azure
19. [生成式 AI 初學者指南](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-122979-leestott) by Microsoft
20. [ChatGPT 初學者指南: 終極使用案例](https://www.udemy.com/course/chatgpt-for-beginners-the-ultimate-use-cases-for-everyone/) by Udemy
21. [[1 小時講座] 大型語言模型入門](https://www.youtube.com/watch?v=zjkBMFhNj_g) by Andrej Karpathy
22. [人人都能使用的 ChatGPT](https://learnprompting.org/courses/chatgpt-for-everyone) by Learn Prompting
23. [大型語言模型 (LLMs) (英語)](https://www.youtube.com/playlist?list=PLxlkzujLkmQ9vMaqfvqyfvZV_o8EqjAk7) by Kshitiz Verma (JK Lakshmipat University, Jaipur, India)

##### 建構 LLM 應用程式

1. [LLMOps: 建構真實世界應用與大型語言模型](https://www.udacity.com/course/building-real-world-applications-with-large-language-models--cd13455) by Udacity

2. [全堆疊 LLM 訓練營](https://fullstackdeeplearning.com/llm-bootcamp/) by FSDL

3. [生成式 AI 初學者指南](https://github.com/microsoft/generative-ai-for-beginners/tree/main) by Microsoft

4. [大型語言模型: 從應用到生產](https://www.edx.org/learn/computer-science/databricks-large-language-models-application-through-production) by Databricks

5. [生成式 AI 基礎](https://www.youtube.com/watch?v=oYm66fHqHUM&list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF) by AWS

6. [生成式 AI 社群課程介紹](https://www.youtube.com/watch?v=ajWheP8ZD70&list=PLmQAMKHKeLZ-iTT-E2kK9uePrJ1Xua9VL) by ineuron

7. [LLM 大學](https://docs.cohere.com/docs/llmu) by Cohere
8. [LLM 學習實驗室](https://lightning.ai/pages/llm-learning-lab/) by Lightning AI
9. [LangChain 用於 LLM 應用開發](https://learn.deeplearning.ai/login?redirect_course=langchain&callbackUrl=https%3A%2F%2Flearn.deeplearning.ai%2Fcourses%2Flangchain) by Deeplearning.AI
10. [LLMOps](https://learn.deeplearning.ai/llmops) by DeepLearning.AI
11. [LLMOps 自動化測試](https://learn.deeplearning.ai/automated-testing-llmops) by DeepLearning.AI
12. [使用 Amazon Bedrock 建構生成式 AI 應用](https://explore.skillbuilder.aws/learn/course/external/view/elearning/17904/building-generative-ai-applications-using-amazon-bedrock-aws-digital-training) by AWS
13. [高效服務 LLMs](https://learn.deeplearning.ai/courses/efficiently-serving-llms/lesson/1/introduction) by DeepLearning.AI
14. [使用 ChatGPT API 建構系統](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) by DeepLearning.AI
15. [使用 Amazon Bedrock 的無伺服器 LLM 應用](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/) by DeepLearning.AI
16. [使用向量資料庫建構應用](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) by DeepLearning.AI
17. [LLMOps 自動化測試](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) by DeepLearning.AI
18. [LLMOps](https://www.deeplearning.ai/short-courses/llmops/) by DeepLearning.AI
19. [使用 LangChain.js 建構 LLM 應用](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/) by DeepLearning.AI
20. [使用 Chroma 的 AI 進階檢索](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) by DeepLearning.AI
21. [在 Azure 上操作 LLMs](https://www.coursera.org/learn/llmops-azure) by Coursera
22. [生成式 AI 全課程 – Gemini Pro, OpenAI, Llama, Langchain, Pinecone, 向量資料庫及更多](https://www.youtube.com/watch?v=mEsleV16qdo) by freeCodeCamp.org
23. [訓練及微調 LLMs 用於生產](https://learn.activeloop.ai/courses/llms) by Activeloop

##### 提示工程、RAG 和微調

1. [LangChain & Vector Databases in Production](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVhnQW8xNDdhSU9IUDVLXzFhV2N0UkNRMkZrQXxBQ3Jtc0traUxHMzZJcGJQYjlyckYxaGxYVWlsOFNGUFlFVEdhNzdjTWpPUlQ2TF9XczRqNkxMVGpJTnd5YmYzV0prQ0IwZURNcHhIZ3h1Z051VTl5MXBBLUN0dkM0NHRkQTFua1Jpc0VCRFJUb0ZQZG95b0JqMA&q=https%3A%2F%2Flearn.activeloop.ai%2Fcourses%2Flangchain&v=gKUTDC13jys) by Activeloop

2. [Reinforcement Learning from Human Feedback](https://learn.deeplearning.ai/reinforcement-learning-from-human-feedback) by DeepLearning.AI

3. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by DeepLearning.AI

4. [Finetuning Large Language Models](https://learn.deeplearning.ai/finetuning-large-language-models) by Deeplearning.AI
5. [LangChain: Chat with Your Data](http://learn.deeplearning.ai/langchain-chat-with-your-data/) by Deeplearning.AI

6. [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system) by Deeplearning.AI
7. [Prompt Engineering with Llama 2](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/) by Deeplearning.AI
8. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by Deeplearning.AI
9. [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction) by Deeplearning.AI
10. [Advanced RAG Orchestration series](https://www.youtube.com/watch?v=CeDS1yvw9E4) by LlamaIndex
11. [Prompt Engineering Specialization](https://www.coursera.org/specializations/prompt-engineering) by Coursera
12. [Augment your LLM Using Retrieval Augmented Generation](https://courses.nvidia.com/courses/course-v1:NVIDIA+S-FX-16+v1/) by Nvidia
13. [Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) by Deeplearning.AI
14. [Open Source Models with Hugging Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/) by Deeplearning.AI
15. [Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) by Deeplearning.AI
16. [Understanding and Applying Text Embeddings](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/) by Deeplearning.AI
17. [JavaScript RAG Web Apps with LlamaIndex](https://www.deeplearning.ai/short-courses/javascript-rag-web-apps-with-llamaindex/) by Deeplearning.AI
18. [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) by Deeplearning.AI
19. [Preprocessing Unstructured Data for LLM Applications](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/) by Deeplearning.AI
20. [Retrieval Augmented Generation for Production with LangChain & LlamaIndex](https://learn.activeloop.ai/courses/rag) by Activeloop
21. [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/) by Deeplearning.AI

##### 評估

1. [建構與評估進階 RAG 應用程式](https://learn.deeplearning.ai/building-evaluating-advanced-rag) by DeepLearning.AI
2. [使用 Weights and Biases 評估與除錯生成式 AI 模型](https://learn.deeplearning.ai/evaluating-debugging-generative-ai) by Deeplearning.AI
3. [LLM 應用程式的品質與安全](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/) by Deeplearning.AI
4. [紅隊測試 LLM 應用程式](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/?utm_campaign=giskard-launch&utm_medium=headband&utm_source=dlai-homepage) by Deeplearning.AI

##### 多模態

1. [擴散模型如何運作](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) by DeepLearning.AI
2. [如何使用 Midjourney、AI 藝術和 ChatGPT 建立一個驚人的網站](https://www.youtube.com/watch?v=5wdCev86RYE) by Brad Hussey
3. [使用 ChatGPT、DALL-E 和 GPT-4 建構 AI 應用程式](https://scrimba.com/learn/buildaiapps) by Scrimba
4. [11-777: 多模態機器學習](https://www.youtube.com/playlist?list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA) by Carnegie Mellon University
5. [視覺模型的提示工程](https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/) by Deeplearning.AI

##### 代理

1. [建構 RAG Agents with LLMs](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-15+V1/) by Nvidia
2. [函式、工具和 Agents with LangChain](https://learn.deeplearning.ai/functions-tools-agents-langchain) by Deeplearning.AI
3. [LangGraph 中的 AI Agents](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) by Deeplearning.AI
4. [使用 AutoGen 的 AI Agentic 設計模式](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/) by Deeplearning.AI
5. [使用 crewAI 的多 AI Agent 系統](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) by Deeplearning.AI
6. [使用 LlamaIndex 建構 Agentic RAG](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) by Deeplearning.AI
7. [LLM 可觀察性: Agents、工具和 Chains](https://courses.arize.com/p/agents-tools-and-chains) by Arize AI

#### 雜項

1. [避免 AI 傷害](https://www.coursera.org/learn/avoiding-ai-harm) by Coursera
2. [制定 AI 政策](https://www.coursera.org/learn/developing-ai-policy) by Coursera

---


## :paperclip: 資源

- [ICLR 2024 論文摘要](https://areganti.notion.site/06f0d4fe46a94d62bff2ae001cfec22c?v=d501ca62e4b745768385d698f173ae14)

---


## :computer: 面試準備

#### 主題相關問題:

1. [常見的生成式 AI 面試問題](interview_prep/60_gen_ai_questions.md)
2. 提示和提示工程
3. 模型微調
4. 模型評估
5. GenAI 的 MLOps
6. 生成模型基礎
7. 最新研究趨勢

#### GenAI 系統設計 (即將推出):

1. 設計一個 LLM 驅動的搜尋引擎
2. 建構一個客戶支援聊天機器人
3. 建構一個與您的資料進行自然語言互動的系統
4. 建構一個 AI 副駕駛
5. 設計一個自訂的聊天機器人，用於多模態資料（文字、圖片、表格、CSV 檔案）的問答
6. 建構一個自動化的產品描述和圖像生成系統，用於電子商務

---


## :notebook: 程式碼筆記本

#### RAG 指南

- [AWS Bedrock Workshop 指南](https://github.com/aws-samples/amazon-bedrock-workshop) by Amazon Web Services
- [Langchain 指南](https://github.com/gkamradt/langchain-tutorials) by gkamradt
- [生產用 LLM 應用](https://github.com/ray-project/llm-applications/tree/main) by ray-project
- [LLM 指南](https://github.com/ollama/ollama/tree/main/examples) by Ollama
- [LLM Hub](https://github.com/mallahyari/llm-hub) by mallahyari

#### 微調指南

- [LLM 微調指南](https://github.com/ashishpatel26/LLM-Finetuning) by ashishpatel26
- [PEFT](https://github.com/huggingface/peft/tree/main/examples) 範例筆記本 by Huggingface
- [免費的 LLM 微調筆記本](https://levelup.gitconnected.com/14-free-large-language-models-fine-tuning-notebooks-532055717cb7) by Youssef Hosni

#### 綜合 LLM 程式碼儲存庫

- [LLM-PlayLab](https://github.com/Sakil786/LLM-PlayLab) 此實驗室涵蓋了通過使用 Transformer 模型製作的多個項目

---


## :black_nib: 貢獻

如果你想新增到這個儲存庫或發現任何問題，請隨時提出 PR，並確保在相關部分或類別中正確放置。

---


## :pushpin: 引用我們

要引用本指南，請使用以下格式:

```
@article{areganti_generative_ai_guide,
author = {Reganti, Aishwarya Naresh},
journal = {https://github.com/aishwaryanr/awesome-generative-ai-resources},
month = {01},
title = {{生成式 AI 指南}}，
year = {2024}
}
```

## 授權條款

[MIT License]

