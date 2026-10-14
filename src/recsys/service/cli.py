"""
src/recsys/service/cli.py
Giao diện dòng lệnh (CLI) cho phép tương tác trực tiếp với RecommendationService.
"""
from __future__ import annotations
import argparse
import sys
from typing import Sequence

from recsys.exceptions import RecSysError
from recsys.service.recommendation_service import RecommendationService


class RecSysCLI:
    """
    Lớp điều khiển giao diện dòng lệnh (CLI Controller).
    Tách biệt hoàn toàn tầng hiển thị (Presentation) với tầng dịch vụ (Service).
    """

    def __init__(self, service: RecommendationService) -> None:
        self._service = service

    def execute_command(self, raw_line: str) -> str:
        """
        Phân tích và thực thi một câu lệnh đơn lẻ trong chế độ tương tác (REPL).
        Hỗ trợ:
          - train
          - recommend <user_id> [top_k]
          - predict <user_id> <movie_id>
          - info
          - help
        """
        tokens = raw_line.strip().split()
        if not tokens:
            return ""

        cmd = tokens[0].lower()

        try:
            if cmd == "help":
                return (
                    "Các lệnh hỗ trợ:\n"
                    "  info                         : Xem trạng thái hệ thống và dữ liệu\n"
                    "  train                        : Huấn luyện mô hình gợi ý\n"
                    "  recommend <user_id> [top_k]  : Lấy danh sách Top-K phim gợi ý cho user\n"
                    "  predict <user_id> <movie_id> : Dự đoán điểm của user cho một movie\n"
                    "  exit / quit                  : Thoát chương trình"
                )

            if cmd == "info":
                summary = self._service.get_system_summary()
                return (
                    f"=== Trạng thái Hệ thống ===\n"
                    f"  Mô hình       : {summary['model_name']}\n"
                    f"  Đã huấn luyện : {summary['is_trained']}\n"
                    f"  Số người dùng : {summary['num_users']}\n"
                    f"  Số bộ phim    : {summary['num_movies']}\n"
                    f"  Số đánh giá   : {summary['num_ratings']}"
                )

            if cmd == "train":
                self._service.train()
                return f"[OK] Đã huấn luyện thành công mô hình {self._service.model_name}."

            if cmd == "recommend":
                if len(tokens) < 2:
                    return "[Lỗi Cú Pháp] Cú pháp đúng: recommend <user_id> [top_k]"
                user_id = self._parse_id(tokens[1])
                top_k = int(tokens[2]) if len(tokens) >= 3 else None

                recs = self._service.recommend(user_id=user_id, top_k=top_k)
                if not recs:
                    return f"Không còn phim mới nào để gợi ý cho User {user_id}."

                lines = [f"=== Top-{len(recs)} Gợi ý cho User '{user_id}' ==="]
                for rec in recs:
                    lines.append(
                        f"  #{rec.rank:<2d} | ID: {str(rec.movie_id):<6s} | "
                        f"Điểm: {rec.score:.4f} | {rec.title}"
                    )
                return "\n".join(lines)

            if cmd == "predict":
                if len(tokens) < 3:
                    return "[Lỗi Cú Pháp] Cú pháp đúng: predict <user_id> <movie_id>"
                user_id = self._parse_id(tokens[1])
                movie_id = self._parse_id(tokens[2])
                score = self._service.predict_score(user_id=user_id, movie_id=movie_id)
                return f"Điểm dự đoán của User '{user_id}' cho Movie '{movie_id}': {score:.4f}"

            return f"[Lỗi] Lệnh không hợp lệ: '{cmd}'. Gõ 'help' để xem hướng dẫn."

        except ValueError:
            return "[Lỗi Tham Số] Tham số top_k phải là một số nguyên hợp lệ."
        except RecSysError as exc:
            # Bắt lỗi đa hình (Polymorphic Exception Handling) cho toàn bộ lỗi nghiệp vụ
            return f"[Lỗi Nghiệp Vụ] {exc}"

    def run_interactive(self) -> None:
        """Chạy vòng lặp giao diện tương tác (REPL) trên terminal."""
        print("====================================================")
        print("  HỆ THỐNG GỢI Ý PHIM HƯỚNG ĐỐI TƯỢNG (OOP RECSYS)  ")
        print("  Gõ 'help' để xem danh sách lệnh, 'exit' để thoát. ")
        print("====================================================")

        while True:
            try:
                line = input("recsys> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nĐã thoát chương trình.")
                break

            if line.lower() in {"exit", "quit"}:
                print("Đã thoát chương trình.")
                break

            output = self.execute_command(line)
            if output:
                print(output)

    @staticmethod
    def _parse_id(raw_id: str) -> int | str:
        """Chuyển đổi ID sang số nguyên nếu có thể, ngược lại giữ nguyên chuỗi."""
        try:
            return int(raw_id)
        except ValueError:
            return raw_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI cho Hệ thống Gợi ý Hướng đối tượng (OOP Recommendation System)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml-100k/u.data",
        help="Đường dẫn tới file dữ liệu đánh giá (ratings).",
    )
    parser.add_argument(
        "--movies-path",
        type=str,
        default="data/ml-100k/u.item",
        help="Đường dẫn tới file thông tin phim (movies).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="popularity",
        choices=["popularity", "matrix_factorization", "bpr"],
        help="Thuật toán gợi ý muốn sử dụng.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="ID người dùng cần gợi ý ngay (nếu không truyền sẽ mở chế độ tương tác).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Số lượng phim gợi ý Top-K.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry-point khi chạy CLI từ terminal."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        from recsys.config import DataConfig, ModelConfig
        from recsys.data.loaders import MovieLens100KLoader

        data_cfg = DataConfig(data_path=args.data_path, movies_path=args.movies_path)
        model_cfg = ModelConfig(model_type=args.model)

        loader = MovieLens100KLoader(
            ratings_path=data_cfg.data_path,
            movies_path=data_cfg.movies_path,
        )
        dataset = loader.load()

        service = RecommendationService.from_config(
            dataset=dataset,
            model_config=model_cfg,
            default_top_k=args.top_k,
        )
        service.train()

        cli = RecSysCLI(service)
        if args.user_id is not None:
            print(cli.execute_command(f"recommend {args.user_id} {args.top_k}"))
        else:
            cli.run_interactive()
        return 0

    except RecSysError as exc:
        print(f"[Lỗi Hệ Thống] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[Lỗi Không Xác Định] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())