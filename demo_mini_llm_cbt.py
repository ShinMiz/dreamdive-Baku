import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

@dataclass
class CBTDocument:
    """CBTナレッジベースの文書"""
    id: str
    content: str
    technique: str
    category: str
    keywords: List[str]

class MiniLLM:
    """極小LLMシミュレーター"""

    def __init__(self):
        # 基本的な応答パターン（FineTuning後の重み）
        self.cbt_weights = {
            "cognitive_restructuring": 0.8,
            "imagery_rehearsal": 0.9,
            "exposure_therapy": 0.7,
            "mindfulness": 0.6,
            "behavioral_activation": 0.5
        }

        # 夢カテゴリの重み
        self.dream_weights = {
            "nightmare": 0.9,
            "recurring": 0.8,
            "anxiety": 0.7,
            "trauma": 0.9,
            "neutral": 0.3
        }

    def generate_response(self, prompt: str, context: List[str], technique: str) -> str:
        """コンテキストを使用した応答生成"""
        base_responses = {
            "cognitive_restructuring": [
                "その考えを別の角度から見てみましょう。{context}について、どう思いますか？",
                "その状況で、他の可能性はありませんか？{context}を参考にしてください。",
                "証拠を整理してみましょう。{context}が示すように、"
            ],
            "imagery_rehearsal": [
                "その夢の結末を変えてみましょう。{context}のように、",
                "より良いバージョンを想像してください。{context}を参考に、",
                "あなたが主人公になれる新しいストーリーを作りましょう。{context}では、"
            ],
            "exposure_therapy": [
                "段階的に慣れていきましょう。{context}が示すように、",
                "小さなステップから始めてください。{context}の方法で、",
                "安全な環境で練習しましょう。{context}を使って、"
            ]
        }

        templates = base_responses.get(technique, ["一般的な応答: {context}"])
        template = random.choice(templates)
        context_str = " ".join(context[:2])  # 最初の2つのコンテキストを使用

        return template.format(context=context_str)

class RAGSystem:
    """RAG（Retrieval-Augmented Generation）システム"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.documents: List[CBTDocument] = []
        self.document_vectors = None
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        """CBT+夢分析のナレッジベース構築"""
        # DiaCBT風のCBTデータ
        cbt_data = [
            {
                "content": "認知再構成法では、自動思考を特定し、その根拠を検証します。破滅的思考パターンを特定し、よりバランスの取れた思考に置き換えます。",
                "technique": "cognitive_restructuring",
                "category": "cbt_technique",
                "keywords": ["認知再構成", "自動思考", "破滅的思考", "バランス"]
            },
            {
                "content": "イメージリハーサル療法は悪夢治療に効果的です。患者は夢の内容を思い出し、より良い結末を想像して練習します。",
                "technique": "imagery_rehearsal",
                "category": "nightmare_treatment",
                "keywords": ["イメージリハーサル", "悪夢", "結末変更", "想像練習"]
            },
            {
                "content": "段階的暴露法では、恐怖刺激に徐々に慣れさせます。不安階層を作成し、最も軽い刺激から始めます。",
                "technique": "exposure_therapy",
                "category": "anxiety_treatment",
                "keywords": ["段階的暴露", "不安階層", "恐怖刺激", "慣れ"]
            }
        ]

        # DreamBank風の夢データ
        dream_data = [
            {
                "content": "追跡される夢は制御感の喪失を表します。追いかけてくるものは避けている問題や感情を象徴することが多いです。",
                "technique": "dream_analysis",
                "category": "chase_dreams",
                "keywords": ["追跡", "制御感", "回避", "問題"]
            },
            {
                "content": "落下する夢は人生の変化への不安を示します。新しい状況への恐れや失敗への心配を反映しています。",
                "technique": "dream_analysis",
                "category": "falling_dreams",
                "keywords": ["落下", "変化", "不安", "失敗"]
            },
            {
                "content": "繰り返し見る夢は未解決の問題を示します。潜在意識が注意を向けようとしているテーマがあります。",
                "technique": "dream_analysis",
                "category": "recurring_dreams",
                "keywords": ["繰り返し", "未解決", "潜在意識", "テーマ"]
            }
        ]

        # 文書オブジェクトの作成
        all_data = cbt_data + dream_data
        for i, doc in enumerate(all_data):
            self.documents.append(CBTDocument(
                id=f"doc_{i}",
                content=doc["content"],
                technique=doc["technique"],
                category=doc["category"],
                keywords=doc["keywords"]
            ))

        # ベクトル化
        contents = [doc.content for doc in self.documents]
        self.document_vectors = self.vectorizer.fit_transform(contents)

    def retrieve(self, query: str, top_k: int = 3) -> List[CBTDocument]:
        """クエリに関連する文書を検索"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        # 類似度でソートして上位k個を取得
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.documents[i] for i in top_indices if similarities[i] > 0.1]

class FineTunedCBTSystem:
    """FineTuningされたCBTシステム"""

    def __init__(self):
        self.llm = MiniLLM()
        self.rag = RAGSystem()
        self.session_history = []

        # FineTuning用のサンプルデータ
        self.training_data = self._create_training_data()
        self._finetune_weights()

    def _create_training_data(self) -> List[Dict]:
        """FineTuning用のトレーニングデータ作成"""
        return [
            {
                "input": "追いかけられる夢を何度も見ます。とても怖いです。",
                "output": "追跡される夢は、避けている問題や感情を表していることが多いです。夢の中で振り返って追いかけてくるものと向き合う練習をしてみましょう。",
                "technique": "imagery_rehearsal",
                "sentiment": "negative"
            },
            {
                "input": "高いところから落ちる夢で目が覚めます。",
                "output": "落下の夢は変化への不安を示します。その夢で、あなたが安全に着地する場面を想像してみてください。",
                "technique": "imagery_rehearsal",
                "sentiment": "anxiety"
            },
            {
                "input": "同じ悪夢を繰り返し見て、眠るのが怖くなりました。",
                "output": "繰り返す悪夢は治療可能です。まず、その夢が伝えようとするメッセージを理解し、結末を変える練習をしましょう。",
                "technique": "cognitive_restructuring",
                "sentiment": "negative"
            }
        ]

    def _finetune_weights(self):
        """簡易的なFineTuning（重み調整）"""
        # トレーニングデータに基づいて重みを調整
        technique_counts = {}
        for data in self.training_data:
            technique = data["technique"]
            technique_counts[technique] = technique_counts.get(technique, 0) + 1

        # 頻出技法の重みを上げる
        total_samples = len(self.training_data)
        for technique, count in technique_counts.items():
            if technique in self.llm.cbt_weights:
                self.llm.cbt_weights[technique] *= (1 + count / total_samples)

    def process_dream(self, dream_text: str) -> Dict:
        """夢の処理とCBT応答生成"""
        # RAGで関連文書を検索
        relevant_docs = self.rag.retrieve(dream_text)

        # コンテキスト作成
        context = [doc.content for doc in relevant_docs]

        # 最適な技法を選択
        technique = self._select_technique(dream_text, relevant_docs)

        # LLMで応答生成
        response = self.llm.generate_response(dream_text, context, technique)

        # セッション履歴に追加
        self.session_history.append({
            "user_input": dream_text,
            "technique": technique,
            "response": response,
            "context_docs": [doc.id for doc in relevant_docs]
        })

        return {
            "response": response,
            "technique": technique,
            "confidence": self._calculate_confidence(relevant_docs),
            "retrieved_docs": [{"id": doc.id, "technique": doc.technique} for doc in relevant_docs],
            "session_count": len(self.session_history)
        }

    def _select_technique(self, dream_text: str, docs: List[CBTDocument]) -> str:
        """最適なCBT技法を選択"""
        # 夢の内容に基づく基本判定
        nightmare_keywords = ["怖い", "恐ろしい", "悪夢", "追いかけ", "落ちる", "死ぬ"]
        is_nightmare = any(keyword in dream_text for keyword in nightmare_keywords)

        # 文書の技法を考慮
        doc_techniques = [doc.technique for doc in docs if doc.technique != "dream_analysis"]

        if is_nightmare and "imagery_rehearsal" in doc_techniques:
            return "imagery_rehearsal"
        elif any(word in dream_text for word in ["思う", "考える", "不安"]) and "cognitive_restructuring" in doc_techniques:
            return "cognitive_restructuring"
        elif doc_techniques:
            return doc_techniques[0]
        else:
            return "imagery_rehearsal"  # デフォルト

    def _calculate_confidence(self, docs: List[CBTDocument]) -> float:
        """応答の信頼度を計算"""
        if not docs:
            return 0.3

        # 検索された文書数と関連度に基づく信頼度
        base_confidence = min(0.9, len(docs) * 0.2 + 0.3)

        # CBT技法の文書があれば信頼度を上げる
        has_cbt_technique = any(doc.technique != "dream_analysis" for doc in docs)
        if has_cbt_technique:
            base_confidence += 0.1

        return round(base_confidence, 2)

# デモ実行
def demo_finetune_cbt():
    """FineTunedCBTシステムのデモ"""
    system = FineTunedCBTSystem()

    test_dreams = [
        "毎晩、何かに追いかけられる夢を見ます。走っても走っても追いつかれそうで、とても怖いです。",
        "高いビルの屋上から落ちる夢を何度も見ます。落ちている間、とても不安になります。",
        "家族が消えてしまう夢を見ました。一人ぼっちになって、とても悲しかったです。",
        "同じ場所で迷子になる夢を繰り返し見ます。出口が見つからなくて焦ります。"
    ]

    print("=== FineTuned CBT LLM + RAG システム デモ ===\n")

    for i, dream in enumerate(test_dreams, 1):
        print(f"【セッション {i}】")
        print(f"患者: {dream}")

        result = system.process_dream(dream)

        print(f"\n🔍 検索された文書: {len(result['retrieved_docs'])}件")
        for doc in result['retrieved_docs']:
            print(f"  - {doc['id']}: {doc['technique']}")

        print(f"\n🧠 選択された技法: {result['technique']}")
        print(f"📊 信頼度: {result['confidence']}")
        print(f"\n💬 CBTシステム応答:")
        print(f"   {result['response']}")

        print(f"\n📈 セッション回数: {result['session_count']}")
        print("\n" + "="*60 + "\n")

    # システム分析
    print("=== システム分析 ===")
    print(f"ナレッジベース文書数: {len(system.rag.documents)}")
    print(f"FineTuning重み:")
    for technique, weight in system.llm.cbt_weights.items():
        print(f"  {technique}: {weight:.2f}")

if __name__ == "__main__":
    demo_finetune_cbt()
