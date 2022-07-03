import re

from markupsafe import escape
from praw import Reddit
from praw.models import Comment

from finder import Displayer


class Bot(Displayer):
    def __init__(self, reddit: Reddit = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reddit = reddit
        self._base_url = 'http://127.0.0.1:5000'
        self._footer = (
            '---',
            'Search for names based on data from the US Social Security Administration |'
            ' [How to use]() |'
            ' [Data sources]()',
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

        if not_found and not reply_lines:
            reply_lines.append('Queries could not be processed.')
        elif not_found:
            reply_lines.append('Remaining queries could not be processed.')

        return reply_lines

    def _query_per_command(self, command: str) -> str:
        _, command_type, query = re.split('!(name|search)\s?', command, 1, re.I)
        if not query:
            pass
        elif command_type == 'name':
            data = self.name(name=query.split(None, 1)[0], n_bars=20)
            if data.get('display'):
                return '\n\n'.join((
                    '**[{name}]({base_url}/n/{name})**'.format(name=data['name'], base_url=self._base_url),
                    '  \n'.join((line for line in data['display']['info'])),
                    self.number_bars_header_text,
                    '  \n'.join((f'    {line}' for line in data['display']['number_bars'])),
                    self.ratio_bars_header_text if data['display']['ratio_bars'] else '',
                    '  \n'.join((f'    {line}' for line in data['display']['ratio_bars'])),
                ))
            return ''
        elif command_type == 'search':
            data = self.search_by_text(query)
            return '\n\n'.join((
                ', '.join('[{name}]({{0}}/n/{name}){display}'.format(**i) for i in data),
                '[Details about your query]({{0}}/q/{query})'.format(query=query),
            )).format(self._base_url)
        return ''


def main():
    bot = Bot()
    bot.load()
    bot.create_reddit()
    bot.run_bot()


if __name__ == '__main__':
    main()
