from modules import prompt_parser, shared


class Parser:
    @staticmethod
    def prompt_to_tags(prompt):
        """
        ポジティブプロンプト文字列からカンマ区切りでタグリストを作成する。
        ※この prompt は既にワイルドカード展開済みの、最終的に画像生成時に使われた文字列である前提です。
        """
        use_prompt_parser = (
            shared.opts.use_prompt_parser_when_save_prompt_to_eagle_as_tags
        )

        p = prompt
        if use_prompt_parser:
            # prompt_parser.parse_prompt_attention は (token, weight) のタプルのリストを返すので token 部分を使用
            p = ",".join(
                map(lambda x: x[0].strip(), prompt_parser.parse_prompt_attention(p))
            )
        return [x.strip() for x in p.split(",") if x.strip() != ""]
