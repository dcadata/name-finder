import re

from markupsafe import escape
from praw import Reddit
from praw.models import Comment

from finder import Displayer


class Bot(Displayer):
    def __init__(self, reddit: Reddit = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reddit = reddit
        self._footer = (
            '---',
            'Search for names based on data from the US Social Security Administration. [How to use]('
            ')',
        )

    def create_reddit(self) -> None:
        self._reddit = Reddit('USNamesBot', user_agent='Search tool for US name data')
        self._reddit.validate_on_submit = True

    def run_bot(self) -> None:
        self._create_multireddit()
        self._monitor_multireddit()

    def _create_multireddit(self) -> None:
        subreddits = {i for i in open('reddit_bot/subreddits.txt').read().splitlines() if not i.startswith('#')}
        self._multireddit = self._reddit.subreddit('+'.join(subreddits))

    def _monitor_multireddit(self) -> None:
        for comment in self._multireddit.stream.comments():
            self._process_request(comment)

    def _process_request(self, request: Comment) -> None:
        if request.saved:
            return
        if message := self._create_reply(escape(request.body)):
            request.save()
            my = request.reply(body=message)
            my.disable_inbox_replies()

    def _create_reply(self, body: str) -> str:
        if reply_lines := self._query_per_request_body(body):
            reply_lines.extend(self._footer)
            return '\n\n'.join(reply_lines)
        return ''

    def _query_per_request_body(self, body: str) -> list:
        raw_commands = [line for line in body.splitlines() if line.startswith(('!name', '!search'))]

        not_found = False
        reply_lines = []
        for raw_command in raw_commands:
            if result := self._query_per_command(raw_command):
                reply_lines.extend((f'> {raw_command}', result))
            else:
                not_found = True

        if not_found:
            reply_lines.append('Remaining queries could not be processed.')

        return reply_lines

    def _query_per_command(self, command: str) -> str:
        _, command_type, query = re.split('!(name|search)\s?', command, 1, re.I)
        if not query:
            pass
        elif command_type == 'name':
            names_ind = query.split(None, 1)[0].split('/')
            if len(names_ind) == 1:
                data = self.name(name=names_ind[0], show_bars=20)
            else:
                data = self.compare(names=tuple(names_ind), show_bars=20)
            return data.get('display', '')
        elif command_type == 'search':
            return self.search_by_text(query).get('display', '')
        return ''


def main():
    bot = Bot()
    bot.load()
    bot.create_reddit()
    bot.run_bot()


if __name__ == '__main__':
    main()
