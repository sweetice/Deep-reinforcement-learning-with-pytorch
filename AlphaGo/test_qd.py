# coding:utf-8

import sys
sys.path.append('../../pythonModules')
import wg4script, AI_QD, wgdensestranet

if __name__ == '__main__':
        num_xd, strategy_id_r, strategy_id_b = 0, 0, 0
        num_plays, num_objcutility = 50, 1
        dic2_rolloutaiparas = {
            'red': {'type_ai': AI_QD.AI_QD_BASE,
                    'type_stra': 'rule-base',
                    'type_stranet': wgdensestranet.StraDenseNet,
                    },
            'blue': {'type_ai': AI_QD.AI_QD_STRA,
                     'type_stra': 'random',
                     'type_stranet': wgdensestranet.StraDenseNet,
                     },
        }
        dic_mainparas = {'str_wgrootdir':'../../',
                         'str_global_flag': 'QD',
                         'num_plays': num_plays,
                         'num_objcutility': num_objcutility,
                         'num_xd': num_xd,
                         'strategy_ids': (strategy_id_r, strategy_id_b),
                         'flag_show': True,
                         'flag_action_cache': False,
                         'flag_qd_rm': True, # flag_qd_rm保存数据库roomrd动作序列
                         'flag_cache': False,
                         'flag_gpu': False,
                         'flag_afm': True, # 两个AI都为BASE时，flag_afm=False; 否则为True
                         'flag_dllnum': 0,
                         'cuda_id': 0,
                         'flag_savestate': False, # flag_savestate： 保存MCTS生成的数据，
                         'dic2_aiparas': {
                             'flag_color4acai': 0,
                             'blue': {'type_ai': AI_QD.AI_QD_HA,
                                     'type_stra': 'rule-base',
                                     # type of stratree of nodes, how to select next path, [rule-base, random, net]
                                     'type_stranet': wgdensestranet.StraDenseNet,
                                     'dic2_rolloutaiparas': dic2_rolloutaiparas,
                                     'flag_candidateactions': 'rule-base'
                                     # [rule-base, stra] how to get candidate actions
                                     },
                             'red': {'type_ai': AI_QD.AI_QD_BASE,
                                      'type_stra': 'net',
                                      'type_stranet': wgdensestranet.StraDenseNet,
                                      'dic2_rolloutaiparas': dic2_rolloutaiparas,
                                      'flag_candidateactions': 'stra'
                                      },
                            },
                         }
        wg4script.simulateframe(dic_mainparas= dic_mainparas)