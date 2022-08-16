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
        self._subreddits = (
            'test', 'lgbt', 'transnames', 'transtryouts', 'agender', 'nonbinary', 'nonbinarytalk', 'genderqueer',
        )
        self._footer = (
            ' | '.join((
                'Search for names based on data from the US Social Security Administration',
                '[How to use]()',
                '[Data sources]()',
            )),
        )

    def run_bot(self) -> None:
        self._reddit.validate_on_submit = True
        multireddit = self._reddit.subreddit('+'.join(self._subreddits))
        for comment in multireddit.stream.comments():
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

    @staticmethod
    def _get_raw_commands(body: str) -> list:
        raw_commands = [line for line in body.splitlines() if line.startswith(('!name', '!search'))]
        return raw_commands

    def _query_per_request_body(self, body: str) -> list:
        not_found = False
        reply_lines = []
        for raw_command in self._get_raw_commands(body):
            if result := self._query_per_command(raw_command):
                reply_lines.extend((f'> {raw_command}', result.format(self._base_url)))
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
            return self._name_query_per_command(query)
        elif command_type == 'search':
            return self._search_query_per_command(query)
        return ''

    def _name_query_per_command(self, query: str) -> str:
        data = self.name(name=query.split(None, 1)[0], n_bars=20)
        if data.get('display'):
            sections = [
                '**[{name}]({{0}}/n/{name})**'.format(name=data['name']),
                '  \n'.join(data['display']['info']),
                self.number_bars_header_text,
                '\n'.join((f'    {line}' for line in data['display']['number_bars'])),
            ]
            if data['display']['ratio_bars']:
                sections.extend([
                    self.ratio_bars_header_text,
                    '\n'.join((f'    {line}' for line in data['display']['ratio_bars'])),
                ])
            return '\n\n'.join(sections)
        return ''

    def _search_query_per_command(self, query: str) -> str:
        data = self.search_by_text(query)
        return '\n\n'.join((
            ', '.join('[{name}]({{0}}/n/{name}){display}'.format(**i) for i in data),
            '[Details about your query]({{0}}/q/{query})'.format(query=query),
        ))
