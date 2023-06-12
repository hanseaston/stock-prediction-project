from data.polygon.API.Resolver import PolygonAPIResolver

if __name__ == '__main__':
    polygon_resolver = PolygonAPIResolver()
    polygon_resolver.resolve_sp500_dataset('2014-01-01', '2019-12-31')
