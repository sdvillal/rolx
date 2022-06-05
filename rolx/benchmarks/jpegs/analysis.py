from __future__ import print_function, division

import os
import os.path as op
import pandas as pd


def collect_results(results_dir=None, do_encoder=False, do_decoder=True):

    if results_dir is None:
        results_dir = op.join(op.dirname(__file__), 'results')

    def explode_encode_result(result):

        tjp_attrs = ['encode_accurate_dct',
                     'encode_fast_dct',
                     'encode_jpeg_subsample',
                     'encode_pixel_format',
                     'encode_quality',
                     'encode_progressive',
                     'name']

        opencv_attrs = ['encode_chroma_quality',
                        'encode_luma_quality',
                        'encode_optimize',
                        'encode_progressive',
                        'encode_quality',
                        'encode_restart_interval',
                        'name']

        keepers = ['name',
                   'encode_accurate_dct',
                   'encode_fast_dct',
                   'encode_jpeg_subsample',
                   'encode_pixel_format',
                   'encode_quality',
                   'encode_optimize']

        for attr in sorted(set(opencv_attrs + tjp_attrs)):
            assert attr not in result
            if attr == 'name':
                result['encoder_name'] = ('opencv' if 'opencv' in result['encoder'].conf.get('name').lower()
                                          else 'turbojpeg')
            else:
                if attr in keepers:
                    try:
                        result[attr] = result['encoder'].conf.get(attr).name
                    except AttributeError:
                        result[attr] = result['encoder'].conf.get(attr)

        result['encoder_id'] = result['encoder'].id()

        del result['encoder']

        return result

    def explode_decode_result(result):

        tjp_attrs = ['decode_accurate_dct',
                     'decode_fast_dct',
                     'decode_fast_upsample',
                     'decode_pixel_format',
                     'decode_scaling_factor',
                     'name']

        opencv_attrs = ['decode_mode',
                        'name']

        keepers = sorted(set(tjp_attrs + opencv_attrs))

        for attr in sorted(set(opencv_attrs + tjp_attrs)):
            assert attr not in result
            if attr == 'name':
                result['decoder_name'] = ('opencv' if 'opencv' in result['decoder'].conf.get('name').lower()
                                          else 'turbojpeg')
            else:
                if attr in keepers:
                    try:
                        result[attr] = result['decoder'].conf.get(attr).name
                    except AttributeError:
                        result[attr] = result['decoder'].conf.get(attr)

        result['decoder_id'] = result['decoder'].id()
        (result['encoded_width'],
         result['encoded_height'],
         result['encoded_subsample'],
         result['encoded_color_space']) = result['encoded_info']

        del result['decoder']
        del result['encoded_info']

        return result

    results = []
    for dirpath, dirnames, filenames in os.walk(results_dir):
        if do_encoder and 'encoder_result.pkl' in filenames:
            result = pd.read_pickle(op.join(dirpath, 'encoder_result.pkl'))
            result = explode_encode_result(result)
            results.append(result)
        if do_decoder and 'decoder_result.pkl' in filenames:
            result = pd.read_pickle(op.join(dirpath, 'decoder_result.pkl'))
            result = explode_encode_result(result)
            result = explode_decode_result(result)
            results.append(result)

    # Dataframe
    df = pd.DataFrame(results)

    # Categoricals
    categoricals = ['encoder_id', 'encoder_name',
                    'decoder_id', 'decoder_name',
                    'turbo_path', 'turbo_version',
                    'host', 'env']
    for categorical in categoricals:
        if categorical in df.columns:
            df[categorical] = df[categorical].astype('category')

    # Useful measures
    df['encode_MB_s'] = (df['uncompressed_size_bytes'] / 1024 ** 2) / df['encode_time_mean_s']
    df['decode_MB_s'] = (df['uncompressed_size_bytes'] / 1024 ** 2) / df['decode_time_mean_s']
    df['space_savings'] = 100 * (1 - df['compressed_size_bytes'] / df['uncompressed_size_bytes'])
    df['phash_ok'] = df['phash'] < 5

    return df


if __name__ == '__main__':

    results_cache = op.join(op.dirname(__file__), 'results', 'decoder.pkl')
    recollect = True

    if recollect or not op.isfile(results_cache):
        ddf = collect_results()
        ddf.to_pickle(results_cache)
    df = pd.read_pickle(results_cache)

    df.info()
    print(df.iloc[0])
