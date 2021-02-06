#!/usr/bin/env python
'''
@author: Leslee
@contact: leelovesc@gmail.com
@time: 2020/3/19 下午8:50
@desc:
'''
import json
def my_result_format(content,entity):

    format_resdate = {
                            "id":None,
                            "title":None,
                            "content":content,
                            "index":None,
                            "pdStructure":None,
                            "pdOrder":None,
                            "parentId":None,
                            "pdPicture":[

                            ],
                            "pdKeyword":[

                            ],
                            "pdAbstract":None,
                            "pdAnnotation":[

                            ],
                            "pdSegment":[

                            ],
                            "pdEntity":entity,
                            "pdRelation":[

                            ],
                            "pdEvent":[

                            ],
                            "pdSentiment":[

                            ],
                            "pdClassification":[

                            ],
                            "pdTranslation":[

                            ],
                            "pdSyntax":{

                            }
                        }


    return format_resdate