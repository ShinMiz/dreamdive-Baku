"""
脳波データ + CBT LLM統合システム
REM睡眠検知結果とCBTシステムを連携
"""

import numpy as np
from demo_mini_llm_cbt import FineTunedCBTSystem
from datetime import datetime, timedelta

class IntegratedDreamAnalysisSystem:
    """統合夢分析システム"""

    def __init__(self):
        self.cbt_system = FineTunedCBTSystem()
        self.sleep_data = {}

    def add_sleep_session(self, session_id: str, rem_data: dict, dream_report: str = None):
        """睡眠セッションデータを追加"""
        self.sleep_data[session_id] = {
            "timestamp": datetime.now(),
            "rem_percentage": rem_data.get("rem_percentage", 0),
            "rem_episodes": rem_data.get("rem_episodes", 0),
            "total_time": rem_data.get("total_time", 0),
            "dream_report": dream_report,
            "analysis_result": None
        }

        # 夢の報告があればCBT分析を実行
        if dream_report:
            analysis = self.cbt_system.process_dream(dream_report)
            self.sleep_data[session_id]["analysis_result"] = analysis

    def generate_sleep_report(self, session_id: str) -> dict:
        """総合睡眠レポート生成"""
        if session_id not in self.sleep_data:
            return {"error": "Session not found"}

        session = self.sleep_data[session_id]

        # REM睡眠の評価
        rem_quality = "良好" if session["rem_percentage"] > 20 else "要注意"

        # CBT推奨事項
        cbt_recommendation = ""
        if session["analysis_result"]:
            technique = session["analysis_result"]["technique"]
            confidence = session["analysis_result"]["confidence"]

            if confidence > 0.7:
                cbt_recommendation = f"{technique}を用いたセラピーを推奨します。"
            else:
                cbt_recommendation = "さらなる夢の記録と分析が必要です。"

        return {
            "session_id": session_id,
            "date": session["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "sleep_metrics": {
                "rem_percentage": session["rem_percentage"],
                "rem_episodes": session["rem_episodes"],
                "total_sleep_time": session["total_time"],
                "rem_quality": rem_quality
            },
            "dream_analysis": session["analysis_result"],
            "recommendations": cbt_recommendation,
            "next_steps": self._generate_next_steps(session)
        }

    def _generate_next_steps(self, session: dict) -> list:
        """次のステップを提案"""
        steps = []

        if session["rem_percentage"] < 15:
            steps.append("REM睡眠が不足しています。睡眠環境の改善を検討してください。")

        if session["dream_report"] and "怖い" in session["dream_report"]:
            steps.append("悪夢に対するイメージリハーサル療法の実践を継続してください。")

        if not session["dream_report"]:
            steps.append("夢日記をつけて、夢の内容を記録してください。")

        return steps

# デモ実行
def demo_integrated_system():
    """統合システムのデモ"""
    system = IntegratedDreamAnalysisSystem()

    # サンプル睡眠データ（demo_REM_analyze.ipynbから想定）
    sample_sessions = [
        {
            "session_id": "sleep_001",
            "rem_data": {
                "rem_percentage": 18.5,
                "rem_episodes": 4,
                "total_time": 480  # 8時間
            },
            "dream_report": "追いかけられる夢を見ました。とても怖くて目が覚めました。"
        },
        {
            "session_id": "sleep_002",
            "rem_data": {
                "rem_percentage": 22.3,
                "rem_episodes": 5,
                "total_time": 420  # 7時間
            },
            "dream_report": "空を飛ぶ夢でした。とても気持ちよかったです。"
        }
    ]

    print("=== 統合夢分析システム デモ ===\n")

    for session_data in sample_sessions:
        # セッションデータを追加
        system.add_sleep_session(
            session_data["session_id"],
            session_data["rem_data"],
            session_data["dream_report"]
        )

        # レポート生成
        report = system.generate_sleep_report(session_data["session_id"])

        print(f"📊 睡眠レポート - {report['session_id']}")
        print(f"日時: {report['date']}")
        print(f"\n🛌 睡眠指標:")
        print(f"  REM睡眠割合: {report['sleep_metrics']['rem_percentage']:.1f}%")
        print(f"  REMエピソード数: {report['sleep_metrics']['rem_episodes']}")
        print(f"  REM睡眠品質: {report['sleep_metrics']['rem_quality']}")

        if report['dream_analysis']:
            print(f"\n🧠 夢分析:")
            print(f"  使用技法: {report['dream_analysis']['technique']}")
            print(f"  信頼度: {report['dream_analysis']['confidence']}")
            print(f"  CBT応答: {report['dream_analysis']['response'][:100]}...")

        print(f"\n💡 推奨事項: {report['recommendations']}")
        print(f"\n📋 次のステップ:")
        for step in report['next_steps']:
            print(f"  • {step}")

        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    demo_integrated_system()
