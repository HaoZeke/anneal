use anneal_core::catalog::{BasinCatalog, Catalog, Event, EventCatalog};

fn one_event() -> Event {
    Event {
        from: 11,
        to: 19,
        dest_energy: -2.5,
    }
}

#[test]
#[allow(deprecated)]
fn the_catalog_alias_preserves_event_catalog_behavior() {
    let mut explicit = EventCatalog::new();
    let mut compatibility = Catalog::new();

    explicit.observe_visit(11);
    compatibility.observe_visit(11);
    assert!(explicit.record_search(11, Some(one_event())));
    assert!(compatibility.record_search(11, Some(one_event())));

    assert_eq!(explicit.event_count(), 1);
    assert_eq!(compatibility.event_count(), explicit.event_count());
    assert_eq!(compatibility.total_searches(), explicit.total_searches());
    assert!(compatibility.known(11, 19));
}

#[test]
fn an_empty_basin_catalog_has_an_explicit_capacity() {
    let catalog = BasinCatalog::with_capacity(32);

    assert_eq!(catalog.capacity(), 32);
    assert_eq!(catalog.len(), 0);
    assert!(catalog.is_empty());
}
