# [第7週] 建構您自己的 LLM 應用程式

## ETMI5: 用五分鐘解釋給我聽

在課程的前幾部分中，我們涵蓋了提示、RAG 和微調等技術，本節將採用實際操作的方法展示如何在應用程式開發中使用 LLM。我們將從基本範例開始，逐步加入更高級的功能，如鏈接、記憶體管理和工具整合。此外，我們還將探討 RAG 和微調的實現。最後，通過整合這些概念，我們將學習如何有效地構建 LLM 代理。

## 簡介

隨著 LLMs 變得越來越普及，現在有多種方式可以利用它們。我們將從基本範例開始，逐步介紹更高級的功能，讓您一步步地在理解的基礎上進行建構。

這份指南旨在涵蓋基礎知識，通過簡單的應用使您熟悉基礎元素。這些範例作為起點，並不適用於生產環境。關於大規模部署應用的見解，包括對 LLM 工具、評估等的討論，請參閱我們前幾週的內容。隨著我們進行每個部分，我們將逐步從基礎組件轉向更高級的組件。

在每個部分中，我們不僅會描述元件，還會提供資源，讓您可以找到程式碼範例來幫助您開發自己的實作。有幾個框架可用於開發您的應用程式，其中一些最知名的包括 LangChain、LlamaIndex、Hugging Face 和 Amazon Bedrock 等。我們的目標是提供來自這些框架廣泛陣列的資源，使您能夠選擇最適合您特定應用程式需求的框架。

當你探索每個部分時，選擇一些資源來幫助使用元件建構應用程式並繼續進行。

![llm_app_steps.png](img/llm_app_steps.png)

## 1. 簡單的 LLM 應用 (提示 + LLM)

**提示:** 在這個上下文中，提示本質上是一個精心構建的請求或指令，用於引導模型生成回應。這是給予 LLM 的初始輸入，概述了你希望它執行的任務或你需要回答的問題。在第二週的內容中，我們深入探討了提示工程，請回到較早的內容以了解更多。

LLM 應用程式開發的基礎方面是使用者定義提示與 LLM 本身之間的互動。這個過程涉及編寫一個明確傳達使用者請求或問題的提示，然後由 LLM 處理以生成回應。例如:

```python
# 定義具有佔位符的提示模板
prompt_template = "提供以下主題的專家建議: {topic}."
# 用實際主題填充模板
prompt = prompt_template.replace("{topic}", topic)
# 呼叫 LLM 的 API
llm_response = call_llm_api(topic)

```

注意，提示作為模板而非固定字串，提升了其在執行時修改的可重用性和靈活性。提示的複雜性可以有所不同；可以根據需求簡單製作或詳細設計。

### 資訊/程式碼

1. [**文件/程式碼**] LangChain Cookbook 用於簡單的 LLM 應用 ([link](https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser))
2. [**影片**] Hugging Face + LangChain in 5 mins by AI Jason ([link](https://www.youtube.com/watch?v=_j7JEDWuqLE))
3. [**文件/程式碼**] 使用 LLMs 與 LlamaIndex ([link](https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms.html))
4. [**部落格**] Leonie Monigatti 的 LangChain 入門 ([link](https://towardsdatascience.com/getting-started-with-langchain-a-beginners-guide-to-building-llm-powered-applications-95fc8898732c))
5. [**筆記本**] LearnDataWithMark 的在自己的筆記型電腦上執行 LLM ([link](https://github.com/mneedham/LearnDataWithMark/blob/main/llm-own-laptop/notebooks/LLMOwnLaptop.ipynb))

---


## 2. 鏈式提示 (提示鏈 + LLM)

雖然使用提示模板和呼叫 LLM 是有效的，有時候你可能需要連續問 LLM 幾個問題，使用之前得到的答案來問下一個問題。想像一下：首先，你問 LLM 你的問題是關於什麼主題。然後，使用該資訊，你請它在該主題上給你一個專家的答案。這種一步一步的過程，其中一個答案引導到下一個問題，稱為「鏈接」。提示鏈基本上就是這個用於執行一系列 LLM 動作的鏈接序列。

LangChain 已成為建構 LLM 應用程式的廣泛使用函式庫，使得能夠將多個問題和答案與 LLM 鏈接起來，產生單一的最終回應。這種方法對於需要多個步驟來達成所需結果的大型專案特別有利。所討論的範例說明了一種基本的鏈接方法。LangChain 的[文件](https://js.langchain.com/docs/modules/chains/)提供了更複雜的鏈接技術指南。

```python
prompt1 ="以下問題是關於什麼主題-{question}?"
prompt2 = "提供關於以下主題的專家建議: {topic}."
```

### 資訊/程式碼

1. **[文章] ****在 Prompt Engineering Guide 上的 Prompt Chaining 文章([link](https://www.promptingguide.ai/techniques/prompt_chaining))
2. [**影片**] 使用 GPT 3.5 和其他 LLMs 的 LLM Chains — LangChain #3 James Briggs ([link](https://www.youtube.com/watch?v=S8j9Tk0lZHU))
3. [**影片**] LangChain 基礎指南 #2 工具和 Chains 由 Sam Witteveen ([link](https://www.youtube.com/watch?v=hI2BY7yl_Ac))
4. [**程式碼**] LangChain 工具和 Chains Colab 筆記本 由 Sam Witteveen ([link](https://colab.research.google.com/drive/1zTTPYk51WvPV8GqFRO18kDe60clKW8VV?usp=sharing))

---


## **3. 添加外部知識庫: 檢索增強生成 (RAG)**

接下來，我們將探索一種不同類型的應用。如果你已經跟隨我們之前的討論，你會知道雖然 LLMs 擅長提供資訊，但他們的知識僅限於最後一次訓練時可用的內容。要生成超出這一點的有意義的輸出，他們需要訪問外部知識庫。這就是檢索增強生成（RAG）所扮演的角色。

檢索增強生成（RAG）就像給你的 LLM 配備了一個個人函式庫來在回答之前進行檢查。在 LLM 想出新東西之前，它會瀏覽一堆資訊（如文章、書籍或網頁）來找到與你的問題相關的內容。然後，它將找到的內容與自身的知識結合起來，給你一個更好的答案。當你需要你的應用程式拉取最新資訊或深入探討特定主題時，這非常方便。

要實現 RAG (Retrieval-Augmented Generation) 超越 LLM 和提示，您將需要以下技術元素:

**知識庫，特別是向量資料庫**

一個全面的文件、文章或數據條目的集合，系統可以依據這些來查找資訊。這個資料庫不僅僅是簡單的文本集合；它通常會轉換成向量資料庫。在這裡，知識庫中的每個項目都會被轉換成代表文本語義意義的高維向量。這種轉換是使用類似於LLM的模型來完成的，但重點是將文本編碼成向量。

建立向量化知識庫的目的是為了實現高效的相似性搜索。當系統嘗試查找與用戶查詢相關的資訊時，它會使用相同的編碼過程將查詢轉換為向量。然後，它會在向量資料庫中搜索最接近查詢向量的向量（即資訊片段），通常使用餘弦相似度等度量方法。這個過程能夠快速識別龐大資料庫中最相關的資訊片段，這在傳統文本搜索方法中是不可行的。

**檢索元件**

檢索元件是執行實際搜索知識庫以查找與使用者查詢相關資訊的引擎。它負責幾個關鍵任務:

1. **查詢編碼:** 它使用與向量化知識庫相同的模型或方法將使用者的查詢轉換為向量。這確保了查詢和資料庫條目處於相同的向量空間中，使相似性比較成為可能。
2. **相似性搜尋:** 一旦查詢被向量化，檢索元件會在向量資料庫中搜尋最接近的向量。此搜尋可以基於各種設計用來有效處理高維數據的演算法，確保過程既快速又準確。
3. **資訊檢索:** 在識別出最接近的向量後，檢索元件會從知識庫中提取相應的條目。這些條目是被認為與使用者查詢最相關的資訊片段。
4. **聚合（可選）:** 在某些實作中，檢索元件還可能會聚合或總結來自多個來源的資訊，以提供綜合回應。這一步在旨在綜合資訊而非直接引用來源的高級RAG系統中更為常見。

在 RAG 框架中，檢索元件的輸出（即檢索到的資訊）會與原始查詢一起輸入到 LLM 中。這使得 LLM 能夠生成不僅在上下文上相關，而且還具有檢索到的資訊的特定性和準確性的回應。結果是一個混合模型，利用了兩者的優點：LLM 的生成靈活性和專用知識庫的事實精確性。

通過將向量化知識庫與高效檢索機制相結合，RAG 系統可以提供既高度相關又由多種來源深入提供的答案。這種方法在需要最新資訊、特定領域知識或超越 LLM 預先知識的詳細解釋的應用中特別有用。

像 LangChain 這樣的框架已經有良好的抽象來建構 RAG 框架

一個來自 LangChain 的簡單範例顯示在[這裡](https://python.langchain.com/docs/expression_language/cookbook/retrieval)。

### 資訊/程式碼

1. [**文章**] 你需要知道的所有內容來建構你的第一個 LLM 應用程式，作者 Dominik Polzer ([link](https://towardsdatascience.com/all-you-need-to-know-to-build-your-first-llm-app-eb982c78ffac))
2. [**影片**] LangChain 的從頭開始 RAG 系列 ([link](https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x))
3. [**影片**] 深入探討使用 LlamaIndex 的檢索增強生成 ([link](https://www.youtube.com/watch?v=Y0FL7BcSigI&t=3s))
4. [**筆記本**] 使用 LangChain 和 Amazon Bedrock Titan 文本及嵌入，使用 OpenSearch 向量引擎筆記本 ([link](https://github.com/aws-samples/rag-using-langchain-amazon-bedrock-and-opensearch))
5. [**影片**] LangChain - 提升檢索效能的高級 RAG 技術，由 Coding Crashcourses 提供 ([link](https://www.youtube.com/watch?v=KQjZ68mToWo))
6. [**影片**] 使用 RAG 的聊天機器人：LangChain 全面指南，由 James Briggs 提供 ([link](https://www.youtube.com/watch?v=LhnCsygAvzY&t=11s))

---


## **4. 增加** 記憶體到 LLMs

我們已經探索了鏈接和整合知識。現在，考慮一個場景，我們需要在與 LLM 的長時間對話中記住過去的互動，其中之前的對話起著作用。

這就是記憶體概念作為重要組成部分發揮作用的地方。記憶體機制，如在 LangChain 等平台上可用的機制，使得對話歷史的儲存成為可能。例如，LangChain 的 ConversationBufferMemory 功能允許保存訊息，這些訊息可以在後續互動中作為上下文檢索和使用。你可以在 LangChain 的[文件](https://python.langchain.com/docs/modules/memory/types/)中發現更多關於這些記憶體抽象及其應用的資訊。

### 資訊/程式碼

1. [**文章**] 使用 LangChain 為 LLMs 提供對話記憶 by Pinecone([link](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/))
2. [**部落格**] 如何為聊天 LLM 模型添加記憶 by Nikolay Penkov ([link](https://medium.com/@penkow/how-to-add-memory-to-a-chat-llm-model-34e024b63e0c))
3. [**文件**] LlamaIndex 文件中的記憶 ([link](https://docs.llamaindex.ai/en/latest/api_reference/memory.html))
4. [**影片**] LangChain: 通過 Prompt Engineering 為 LLMs 提供記憶 ([link](https://www.youtube.com/watch?v=dxO6pzlgJiY))
5. [**影片**] 建構具有記憶的 LangChain 自訂醫療代理 ([link](https://www.youtube.com/watch?v=6UFtRwWnHws))

---


## **5. 使用外部工具與 LLMs**

考慮在 LLM 應用程式中的一個場景，例如旅行規劃器，其中目的地或景點的可用性取決於季節性開放。想像我們可以訪問一個提供此特定資訊的 API。在這種情況下，應用程式必須查詢 API 以確定某個地點是否開放。如果地點關閉，LLM 應該相應地調整其建議，提出替代選項。這說明了一個關鍵實例，即整合外部工具可以顯著增強 LLM 的功能，使其能夠提供更準確和上下文相關的回應。這種整合不僅限於旅行規劃；還有許多其他情況，其中外部資料來源、API 和工具可以豐富 LLM 應用程式。範例包括用於活動規劃的天氣預報、用於財務建議的股市資料或用於內容產生的即時新聞，每一個都為 LLM 的能力增加了一層動態性和特異性。

在像 LangChain 這樣的框架中，通過其鏈接框架整合這些外部工具，使得新元素（如 API、資料來源和其他工具）的無縫整合成為可能。

### 資訊/程式碼

1. [**文件/程式碼**] LangChain 的 LLM 工具列表 ([link](https://python.langchain.com/docs/integrations/tools))
2. [**文件/程式碼**] LlamaIndex 的工具 ([link](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html))
3. [**影片**] Sam Witteveen 的 LangChain 自訂工具和代理建構 ([link](https://www.youtube.com/watch?v=biS8G8x8DdA))

---


## **6. LLMs 做決策: Agents**

在前面的部分中，我們探討了複雜的 LLM 元件，如工具和記憶體。現在，假設我們希望 LLM 能夠有效地利用這些元素來替我們做決策。

LLM 代理正是這樣做的，它們是透過將 LLM 與其他模組（如計劃、記憶體和工具使用）相結合來執行複雜任務的系統。這些代理利用 LLM 的能力來理解和生成類似人類的語言，使它們能夠與使用者互動並有效地處理資訊。

例如，考慮一個情境，我們希望 LLM 代理協助財務規劃。任務是分析個人在過去一年的消費習慣，並提供預算最佳化的建議。

為了完成這項任務，代理首先利用其記憶體模組來存取有關個人支出、收入來源和財務目標的儲存資料。然後，它使用規劃機制將任務分解成幾個步驟:

1. **資料分析**: 該代理使用外部工具處理財務資料，分類支出、識別趨勢並計算關鍵指標，如總支出、儲蓄率和支出分佈。
2. **預算評估**: 根據分析的資料，LLM代理評估當前預算在實現個人財務目標方面的有效性。它考慮了可自由支配的支出、基本開支和潛在的成本削減領域。
3. **建議生成**: 利用其對財務原則和最佳化策略的理解，代理制定個性化建議以改善個人的財務健康狀況。這些建議可能包括將資金重新分配到儲蓄、減少非必要支出或探索投資機會。
4. **溝通**: 最後，LLM代理以清晰易懂的方式向用戶傳達建議，使用自然語言生成能力解釋每個建議背後的理由和潛在的好處。

在整個過程中，LLM代理無縫整合其決策能力與外部工具、記憶體儲存和規劃機制，以提供針對用戶財務狀況的可行見解。

以下是 LLM 代理如何結合各種元件來做出決策的方法:

1. **語言模型 (LLM)**: LLM 作為代理的中央控制器或「大腦」。它解釋用戶查詢、生成回應，並協調完成任務所需的整體操作流程。
2. **關鍵模組**:
    - **規劃**: 此模組幫助代理將複雜任務分解為可管理的子部分。它制定行動計劃，以有效地達成所需目標。
    - **記憶體**: 記憶體模組允許代理存儲和檢索與當前任務相關的資訊。它有助於維持操作狀態、跟蹤進度，並根據過去的觀察做出明智的決策。
    - **工具使用**: 代理可能利用外部工具或 API 來收集數據、執行計算或生成輸出。與這些工具的整合增強了代理解決各種任務的能力。

現有的框架提供內建模組和抽象來建構代理。請參考下面提供的資源來實作您自己的代理。

### 資訊/程式碼

1. [**文件/程式碼**] LangChain 中的代理 ([link](https://python.langchain.com/docs/modules/agents/))
2. [**文件/程式碼**] LlamaIndex 中的代理 ([link](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/root.html))
3. [**影片**] LangChain 代理 - Sam Witteveen 的工具和鏈接決策 ([link](https://www.youtube.com/watch?v=ziu87EXZVUE&t=59s))
4. [**文章**] Nvidia 建構您的第一個 LLM 代理應用 ([link](https://developer.nvidia.com/blog/building-your-first-llm-agent-application))
5. [**影片**] OpenAI 函式 + LangChain : Sam Witteveen 的多工具代理建構 ([link](https://www.youtube.com/watch?v=4KXK6c6TVXQ))

---


## **7. 微調**

在前面的部分中，我們探討了使用預訓練的 LLM 與額外元件。然而，有些情況下必須在使用前更新 LLM 以包含相關資訊，特別是當 LLM 缺乏對某個主題的特定知識時。在這種情況下，有必要先微調 LLM，然後再應用第 1-5 部分中概述的策略來建構應用程式。

各種平台提供微調功能，但需要注意的是，微調比僅僅從LLM獲取回應需要更多的資源，因為它涉及訓練模型以理解和生成所需主題的資訊。

### 資訊/程式碼

1. [**文章**] 2024 年如何使用 Hugging Face 微調 LLMs by philschmid ([link](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl))
2. [**影片**] 微調大型語言模型 (LLMs) | 含範例程式碼 by Shaw Talebi ([link](https://www.youtube.com/watch?v=eC6Hd1hFvos))
3. [**影片**] 使用 PEFT 和 LoRA 微調 LLMs by Sam Witteveen ([link](https://www.youtube.com/watch?v=Us5ZFp16PaU&t=261s))
4. [**影片**] LLM 微調速成課程: 一小時端到端指南 by AI Anytime ([link](https://www.youtube.com/watch?v=mrKuDK9dGlg))
5. [**文章**] Weights and Biases 的 LLM 微調系列 ([link](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2))。

---


## 閱讀/觀看這些資源 (選擇性)

1. aishwaryanr 的 LLM 筆記本清單 ([link](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#notebook-code-notebooks))
2. Sam Witteveen 的 LangChain 如何做和指南 ([link](https://www.youtube.com/watch?v=J_0qvRt4LNk&list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ))
3. codebasics 的 LangChain 初學者速成課程 | LangChain 指南 ([link](https://www.youtube.com/watch?v=nAmC7SoVLd8))
4. LangChain 系列建構 ([link](https://www.youtube.com/watch?v=mmBo8nlu2j0&list=PLfaIDFEXuae06tclDATrMYY0idsTdLg9v))
5. Maxime Labonne 的 LLM 實作課程 ([link](https://github.com/mlabonne/llm-course))

