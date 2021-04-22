#!/usr/bin/env python
'''
@author: Leslee
@contact: leelovesc@gmail.com
@time: 2020/3/19 下午8:50
@desc:
'''


def my_result_format(content, pd_classify):
    format_resdate = {
                        "id": None,
                        "title": None,
                        "content": content,
                        "index": None,
                        "pdStructure": None,
                        "pdOrder": None,
                        "parentId": None,
                        "pdPicture": [
                        ],
                        "pdKeyword": [
                        ],
                        "pdAbstract": None,
                        "pdAnnotation": [
                        ],
                        "pdSegment": [
                        ],
                        "pdEntity": [],
                        "pdRelation": [
                        ],
                        "pdEvent": [
                        ],
                        "pdSentiment": [
                        ],
                        "pdClassification": pd_classify,
                        "pdTranslation": [
                        ],
                        "pdSyntax": {
                        }
                    }
    return format_resdate
